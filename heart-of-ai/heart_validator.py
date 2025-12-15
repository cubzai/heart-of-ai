"""
HEART Validator Module
Implements learned epistemic policies with discrete actions:
- accept: Continue generation
- abstain: Return uncertainty
- retrieve: Trigger RAG
- repair: Self-correction
"""

import tensorflow as tf
import numpy as np


class SparseAutoencoder(tf.keras.layers.Layer):
    """
    Sparse Autoencoder for mechanistic interpretability
    Extracts monosemantic features from LM activations
    """
    def __init__(self, input_dim, feature_dim, sparsity=0.1):
        super(SparseAutoencoder, self).__init__()
        self.input_dim = input_dim
        self.feature_dim = feature_dim
        self.sparsity = sparsity
        
        self.encoder = tf.keras.layers.Dense(feature_dim, activation='relu')
        self.decoder = tf.keras.layers.Dense(input_dim)
        self.bias_decoder = tf.Variable(tf.zeros([input_dim]))
        
    def call(self, x, training=False):
        """
        Args:
            x: Input activations [batch, input_dim]
        Returns:
            features: Sparse features [batch, feature_dim]
            reconstructed: Reconstruction [batch, input_dim]
            loss: Sparsity + reconstruction loss
        """
        # Encode
        h = self.encoder(x)
        features = h
        
        # Enforce sparsity via L1 regularization
        sparsity_loss = tf.reduce_mean(tf.abs(features)) * self.sparsity
        
        # Decode with bias
        reconstructed = self.decoder(features) + self.bias_decoder
        
        # Reconstruction loss
        reconstruction_loss = tf.reduce_mean(tf.square(x - reconstructed))
        
        total_loss = reconstruction_loss + sparsity_loss
        
        return features, reconstructed, total_loss


class ValidatorNetwork(tf.keras.Model):
    """
    Epistemic Validator Network
    Maps token/concept signals to discrete actions
    """
    def __init__(self, hidden_dim=512, n_actions=4):
        super(ValidatorNetwork, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_actions = n_actions  # {accept, abstain, retrieve, repair}
        
        # Token-level SAE for interpretability
        self.token_sae = SparseAutoencoder(input_dim=768, feature_dim=256)
        
        # Concept-level processing
        self.concept_fc1 = tf.keras.layers.Dense(hidden_dim, activation='relu')
        self.concept_fc2 = tf.keras.layers.Dense(hidden_dim // 2, activation='relu')
        
        # Alignment score branch
        self.alignment_fc = tf.keras.layers.Dense(1, activation='sigmoid')
        
        # Action policy branch
        self.action_fc1 = tf.keras.layers.Dense(hidden_dim, activation='relu')
        self.action_fc2 = tf.keras.layers.Dense(n_actions)
        
        # Attention for alignment computation
        self.attention_layer = tf.keras.layers.MultiHeadAttention(
            num_heads=4, key_dim=64
        )
    
    def call(self, token_logits, token_embeddings, h_l, h_h, 
             attention_patterns=None, training=False):
        """
        Args:
            token_logits: LM logits [batch, seq, vocab]
            token_embeddings: Token embeddings [batch, seq, embd]
            h_l: L-module state [batch, d_l]
            h_h: H-module state [batch, d_h]
            attention_patterns: Attention weights [batch, heads, seq, seq]
            
        Returns:
            alignment_score: α_t ∈ [0,1]
            action_logits: [batch, 4]
            interpretability_features: Dict with SAE features
        """
        batch_size = tf.shape(token_embeddings)[0]
        seq_len = tf.shape(token_embeddings)[1]
        
        # ===== Token-level signal processing =====
        # Extract SAE features for mechanistic interpretability
        flattened_emb = tf.reshape(token_embeddings, [-1, 768])
        sae_features, _, sae_loss = self.token_sae(flattened_emb, training=training)
        sae_features = tf.reshape(sae_features, [batch_size, seq_len, -1])
        
        # Pool SAE features: identify key concept activations
        concept_activations = tf.reduce_mean(sae_features, axis=1)  # [batch, 256]
        
        # Compute confidence from logits
        logit_probs = tf.nn.softmax(token_logits, axis=-1)
        logit_entropy = -tf.reduce_sum(logit_probs * tf.math.log(logit_probs + 1e-10), axis=-1)
        entropy_score = tf.reduce_mean(logit_entropy, axis=1)  # [batch]
        
        # ===== Alignment score computation =====
        # Combine multiple signals: SAE + entropy + L/H states
        alignment_input = tf.concat([
            concept_activations,           # SAE features
            tf.expand_dims(entropy_score, 1) * tf.ones([batch_size, 256]),  # Broadcast entropy
            h_l,                           # L-module context
            h_h[:, :256],                  # H-module context (slice if needed)
        ], axis=1)
        
        alignment_score = self.alignment_fc(alignment_input)  # [batch, 1]
        alignment_score = tf.squeeze(alignment_score, axis=-1)  # [batch]
        
        # ===== Action policy computation =====
        # Concept-level reasoning
        concept_hidden = self.concept_fc1(concept_activations)
        concept_hidden = self.concept_fc2(concept_hidden)
        
        # Combine with module states
        action_input = tf.concat([
            concept_hidden,
            h_l,
            h_h,
        ], axis=1)
        
        action_hidden = self.action_fc1(action_input)
        action_logits = self.action_fc2(action_hidden)  # [batch, 4]
        
        # ===== Interpretability output =====
        interpretability = {
            'sae_features': sae_features,
            'concept_activations': concept_activations,
            'logit_entropy': entropy_score,
            'alignment_score': alignment_score,
            'action_logits': action_logits,
            'sae_loss': sae_loss
        }
        
        return alignment_score, action_logits, interpretability


class EpistemicActionExecutor:
    """
    Executes epistemic actions:
    1. ACCEPT - Continue generation
    2. ABSTAIN - Return uncertainty statement
    3. RETRIEVE - Trigger RAG
    4. REPAIR - Self-correct
    """
    
    ACTION_ACCEPT = 0
    ACTION_ABSTAIN = 1
    ACTION_RETRIEVE = 2
    ACTION_REPAIR = 3
    
    ACTION_NAMES = {
        0: "ACCEPT",
        1: "ABSTAIN",
        2: "RETRIEVE",
        3: "REPAIR"
    }
    
    def __init__(self, validator_network, retrieval_fn=None, repair_fn=None):
        self.validator = validator_network
        self.retrieval_fn = retrieval_fn  # RAG function
        self.repair_fn = repair_fn         # Self-correction function
        
    def execute(self, action_idx, generated_tokens, alignment_score, 
                context=None, retrieved_docs=None):
        """
        Execute epistemic action
        
        Args:
            action_idx: Action index 0-3
            generated_tokens: Currently generated tokens
            alignment_score: Confidence from validator
            context: Generation context
            retrieved_docs: Retrieved documents (if any)
            
        Returns:
            result: Dict with action result
        """
        action_name = self.ACTION_NAMES[action_idx]
        
        if action_idx == self.ACTION_ACCEPT:
            return self._accept(generated_tokens, alignment_score)
        
        elif action_idx == self.ACTION_ABSTAIN:
            return self._abstain(alignment_score)
        
        elif action_idx == self.ACTION_RETRIEVE:
            return self._retrieve(generated_tokens, context)
        
        elif action_idx == self.ACTION_REPAIR:
            return self._repair(generated_tokens, context, alignment_score)
        
        else:
            raise ValueError(f"Unknown action: {action_idx}")
    
    def _accept(self, tokens, alignment_score):
        """Continue generation - commit tokens to output"""
        return {
            'action': 'ACCEPT',
            'tokens': tokens,
            'alignment': alignment_score,
            'committed': True
        }
    
    def _abstain(self, alignment_score):
        """Halt and return uncertainty statement"""
        uncertainty_statement = (
            "I'm not confident about my response. "
            "I cannot provide a reliable answer to this query."
        )
        return {
            'action': 'ABSTAIN',
            'tokens': [],
            'statement': uncertainty_statement,
            'confidence': alignment_score,
            'committed': False
        }
    
    def _retrieve(self, tokens, context):
        """Trigger RAG to retrieve relevant documents"""
        result = {
            'action': 'RETRIEVE',
            'tokens': tokens,
            'context': context,
            'requires_external_knowledge': True
        }
        
        if self.retrieval_fn is not None:
            retrieved = self.retrieval_fn(context)
            result['retrieved_documents'] = retrieved
        
        return result
    
    def _repair(self, tokens, context, alignment_score):
        """Self-correction cycle"""
        result = {
            'action': 'REPAIR',
            'original_tokens': tokens,
            'context': context,
            'repair_confidence': alignment_score,
            'requires_regen': True
        }
        
        if self.repair_fn is not None:
            repaired = self.repair_fn(tokens, context)
            result['repaired_tokens'] = repaired
        
        return result


class ValidatorTrainer:
    """
    Training utilities for the epistemic validator
    """
    def __init__(self, validator_network, learning_rate=1e-3):
        self.validator = validator_network
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    
    def compute_losses(self, alignment_scores, action_logits, 
                      target_alignments, target_actions, 
                      alignment_weights=0.5):
        """
        Compute validator training losses
        
        Loss = alignment_loss + epistemic_loss + auxiliary_losses
        """
        # 1. Alignment Loss: MSE between predicted and empirical correctness
        alignment_loss = tf.reduce_mean(
            tf.square(alignment_scores - target_alignments)
        )
        
        # 2. Epistemic Classification Loss: Cross-entropy for action prediction
        epistemic_loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=target_actions,
                logits=action_logits
            )
        )
        
        # 3. Auxiliary Task: Predict if retrieval will help
        # (would need auxiliary labels in full implementation)
        
        # Combined loss
        total_loss = (alignment_weights * alignment_loss + 
                     (1 - alignment_weights) * epistemic_loss)
        
        return {
            'total_loss': total_loss,
            'alignment_loss': alignment_loss,
            'epistemic_loss': epistemic_loss
        }
    
    def train_step(self, batch_data):
        """
        Single training step
        
        batch_data should contain:
        - token_logits
        - token_embeddings
        - h_l, h_h
        - target_alignments
        - target_actions
        """
        with tf.GradientTape() as tape:
            alignment_scores, action_logits, _ = self.validator(
                batch_data['token_logits'],
                batch_data['token_embeddings'],
                batch_data['h_l'],
                batch_data['h_h'],
                training=True
            )
            
            losses = self.compute_losses(
                alignment_scores,
                action_logits,
                batch_data['target_alignments'],
                batch_data['target_actions']
            )
        
        # Backprop
        trainable_vars = self.validator.trainable_variables
        gradients = tape.gradient(losses['total_loss'], trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        
        return losses


class ValidatorInterpretability:
    """
    Mechanistic interpretability analysis of validator decisions
    via Sparse Autoencoder feature circuits
    """
    def __init__(self, validator_network):
        self.validator = validator_network
    
    def identify_feature_circuits(self, interpretability_features, action_taken):
        """
        Identify which SAE features triggered specific actions
        
        Returns:
        - abstain_features: Features associated with abstain action
        - retrieve_features: Features associated with retrieve action
        - repair_features: Features associated with repair action
        """
        sae_features = interpretability_features['sae_features']  # [batch, seq, 256]
        
        # Pool across sequence
        feature_importance = tf.reduce_mean(sae_features, axis=1)  # [batch, 256]
        
        # Get top features for this action
        top_k = 10
        top_indices = tf.nn.top_k(feature_importance[0], k=top_k)[1]
        
        return {
            'top_features': top_indices.numpy(),
            'feature_activations': feature_importance,
            'action_taken': action_taken
        }
    
    def explain_decision(self, alignment_score, action_logits, 
                        interpretability_features):
        """
        Generate human-readable explanation of validator decision
        """
        action_idx = tf.argmax(action_logits, axis=-1)[0].numpy()
        action_name = EpistemicActionExecutor.ACTION_NAMES[action_idx]
        
        explanation = {
            'action': action_name,
            'alignment_confidence': float(alignment_score[0].numpy()),
            'action_probabilities': tf.nn.softmax(action_logits, axis=-1)[0].numpy(),
            'key_features_active': self.identify_feature_circuits(
                interpretability_features, action_idx
            )
        }
        
        return explanation


if __name__ == "__main__":
    # Test validator network
    validator = ValidatorNetwork(hidden_dim=512, n_actions=4)
    
    # Create dummy inputs
    batch_size = 2
    seq_len = 128
    token_logits = tf.random.normal([batch_size, seq_len, 50257])
    token_embeddings = tf.random.normal([batch_size, seq_len, 768])
    h_l = tf.random.normal([batch_size, 256])
    h_h = tf.random.normal([batch_size, 512])
    
    # Forward pass
    alignment, actions, interp = validator(
        token_logits, token_embeddings, h_l, h_h, training=False
    )
    
    print(f"Alignment shape: {alignment.shape}")
    print(f"Action logits shape: {actions.shape}")
    print(f"Action probabilities: {tf.nn.softmax(actions[0]).numpy()}")
    print("\nValidator network created successfully!")