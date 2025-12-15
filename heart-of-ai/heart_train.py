"""
HEART Training Module
Implements end-to-end training pipeline with:
- Hierarchical supervision
- Epistemic validator training
- Continual learning mechanisms
"""

import tensorflow as tf
import numpy as np
from heart_model import HEART, HeartConfig
from heart_validator import ValidatorTrainer


class HEARTTrainer:
    """
    Complete training pipeline for HEART architecture
    """
    def __init__(self, config, base_model_fn, validator_fn=None):
        self.config = config
        self.base_model_fn = base_model_fn
        
        # Create HEART model
        self.heart = HEART(config, base_model_fn)
        
        # Create validator trainer if not provided
        if validator_fn is None:
            from heart_validator import ValidatorNetwork
            validator_fn = ValidatorNetwork(hidden_dim=512, n_actions=4)
        
        self.validator_trainer = ValidatorTrainer(
            validator_fn, 
            learning_rate=config.learning_rate
        )
        
        # Optimizers
        self.heart_optimizer = tf.keras.optimizers.Adam(
            learning_rate=config.learning_rate
        )
        self.validator_optimizer = tf.keras.optimizers.Adam(
            learning_rate=config.learning_rate
        )
        
        # Metrics
        self.metrics = {
            'generation_loss': tf.keras.metrics.Mean(),
            'alignment_loss': tf.keras.metrics.Mean(),
            'epistemic_loss': tf.keras.metrics.Mean(),
            'total_loss': tf.keras.metrics.Mean(),
            'hallucination_error': tf.keras.metrics.Mean(),
            'validator_accuracy': tf.keras.metrics.Mean(),
        }
    
    def train_step(self, batch):
        """
        Single training step
        
        batch should contain:
        - input_ids: [batch, seq_len]
        - target_ids: [batch, seq_len]
        - alignment_labels: [batch, seq_len] (0/1)
        - action_labels: [batch, seq_len] (0-3)
        """
        with tf.GradientTape() as tape:
            # Forward pass
            output, validator_info = self.heart(
                batch['input_ids'],
                training=True,
                return_validator_info=True
            )
            
            # Compute losses
            losses = self.heart.compute_losses(
                output,
                batch['target_ids'],
                validator_info['validator_actions'],
                training=True
            )
            
            # Additional validator losses
            validator_losses = self._compute_validator_losses(
                validator_info,
                batch.get('action_labels'),
                batch.get('alignment_labels')
            )
            
            # Combined loss
            total_loss = (losses['total_loss'] + 
                         0.5 * validator_losses['validator_total_loss'])
        
        # Update HEART
        heart_vars = self.heart.trainable_variables
        gradients = tape.gradient(total_loss, heart_vars)
        self.heart_optimizer.apply_gradients(zip(gradients, heart_vars))
        
        # Update metrics
        self.metrics['generation_loss'].update_state(losses['generation_loss'])
        self.metrics['alignment_loss'].update_state(losses['alignment_loss'])
        self.metrics['epistemic_loss'].update_state(losses['epistemic_loss'])
        self.metrics['total_loss'].update_state(total_loss)
        
        return {
            'total_loss': float(total_loss.numpy()),
            'generation_loss': float(losses['generation_loss'].numpy()),
            'alignment_loss': float(losses['alignment_loss'].numpy()),
            'epistemic_loss': float(losses['epistemic_loss'].numpy()),
            **{k: float(v.numpy()) for k, v in validator_losses.items()}
        }
    
    def _compute_validator_losses(self, validator_info, action_labels=None, 
                                 alignment_labels=None):
        """Compute validator-specific losses"""
        losses = {}
        
        # Alignment loss from validator signals
        alignment_scores = validator_info.get('alignment_scores', [])
        if alignment_scores and alignment_labels is not None:
            alignment_loss = tf.reduce_mean(
                tf.square(alignment_scores - alignment_labels)
            )
            losses['alignment_calibration_loss'] = alignment_loss
        
        # Action classification loss
        action_logits_list = validator_info.get('action_logits', [])
        if action_logits_list and action_labels is not None:
            action_loss = 0.0
            for logits in action_logits_list:
                action_loss += tf.reduce_mean(
                    tf.nn.sparse_softmax_cross_entropy_with_logits(
                        labels=tf.argmax(action_labels, axis=-1),
                        logits=logits
                    )
                )
            action_loss /= len(action_logits_list)
            losses['action_classification_loss'] = action_loss
        
        # Total validator loss
        losses['validator_total_loss'] = sum(losses.values()) if losses else tf.constant(0.0)
        
        return losses
    
    def train_epoch(self, dataset, steps_per_epoch=None):
        """Train for one epoch"""
        epoch_losses = {k: [] for k in self.metrics.keys()}
        
        for step, batch in enumerate(dataset):
            if steps_per_epoch and step >= steps_per_epoch:
                break
            
            losses = self.train_step(batch)
            
            for key in epoch_losses.keys():
                if key in losses:
                    epoch_losses[key].append(losses[key])
            
            if (step + 1) % 10 == 0:
                print(f"Step {step + 1}: Loss={losses['total_loss']:.4f}")
        
        # Compute epoch averages
        epoch_summary = {}
        for key, values in epoch_losses.items():
            if values:
                epoch_summary[key] = np.mean(values)
        
        return epoch_summary
    
    def validate(self, val_dataset, num_steps=None):
        """Validation step"""
        val_losses = []
        
        for step, batch in enumerate(val_dataset):
            if num_steps and step >= num_steps:
                break
            
            # Forward pass (no gradients)
            output, _ = self.heart(batch['input_ids'], training=False, 
                                  return_validator_info=True)
            
            # Compute loss
            losses = self.heart.compute_losses(
                output, batch['target_ids'], [], training=False
            )
            val_losses.append(float(losses['total_loss'].numpy()))
        
        return np.mean(val_losses) if val_losses else float('inf')


class HeartCLTrainer:
    """
    HEART-CL: Continual Learning trainer for online adaptation
    """
    def __init__(self, heart_model, config):
        self.heart = heart_model
        self.config = config
        self.adaptation_history = []
        
    def adapt_on_feedback(self, generated_tokens, correct_tokens, 
                         epistemic_signals, learning_rate=1e-6):
        """
        Adapt HEART parameters based on correction feedback
        
        Args:
            generated_tokens: What HEART generated
            correct_tokens: Ground truth
            epistemic_signals: Validator signals (abstain, retrieve, etc)
            learning_rate: Very low for stability
            
        Returns:
            adaptation_info: Dict with adaptation metrics
        """
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        
        with tf.GradientTape() as tape:
            # Reconstruction loss
            repair_loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=correct_tokens,
                    logits=tf.ones_like(generated_tokens)
                )
            )
            
            # Consistency loss (stay close to pre-training)
            consistency_loss = 0.0  # Would compare to original behavior
            
            # Total adaptation loss
            total_loss = repair_loss + 0.5 * consistency_loss
        
        # Update L/H modules
        trainable_vars = (self.heart.l_module.trainable_variables + 
                         self.heart.h_module.trainable_variables)
        
        gradients = tape.gradient(total_loss, trainable_vars)
        optimizer.apply_gradients(zip(gradients, trainable_vars))
        
        adaptation_info = {
            'repair_loss': float(repair_loss.numpy()),
            'consistency_loss': float(consistency_loss),
            'total_adaptation_loss': float(total_loss.numpy()),
            'timestamp': len(self.adaptation_history)
        }
        
        self.adaptation_history.append(adaptation_info)
        
        return adaptation_info


class DistributedHEARTTrainer:
    """
    Multi-GPU/TPU training for HEART
    """
    def __init__(self, config, base_model_fn, strategy):
        self.config = config
        self.base_model_fn = base_model_fn
        self.strategy = strategy
        
        with strategy.scope():
            self.trainer = HEARTTrainer(config, base_model_fn)
    
    def train_step_distributed(self, batch):
        """Distributed training step"""
        
        @tf.function
        def distributed_train_step(batch):
            per_replica_losses = self.strategy.run(
                self.trainer.train_step, 
                args=(batch,)
            )
            return self.strategy.reduce(
                tf.distribute.ReduceOp.MEAN, 
                per_replica_losses, 
                axis=None
            )
        
        return distributed_train_step(batch)


class HEARTBenchmark:
    """
    Evaluate HEART on standard benchmarks
    """
    def __init__(self, heart_model, validator, encoder):
        self.heart = heart_model
        self.validator = validator
        self.encoder = encoder
    
    def evaluate_hallucination(self, dataset):
        """
        Evaluate on hallucination benchmarks:
        - TruthfulQA
        - HaluEval
        - CRAG
        """
        hallucination_scores = []
        
        for question, ground_truth in dataset:
            # Generate with HEART
            prompt_tokens = self.encoder.encode(question)
            output, validator_info = self.heart(
                tf.constant([prompt_tokens]),
                training=False,
                return_validator_info=True
            )
            
            generated_text = self.encoder.decode(output[0].numpy())
            
            # Check factual correctness (simplified)
            is_correct = self._check_correctness(generated_text, ground_truth)
            
            # Get validator confidence
            validator_confidence = np.mean([
                a['alignment'] for a in validator_info.get('validator_actions', [])
            ])
            
            hallucination_scores.append({
                'correct': is_correct,
                'validator_confidence': validator_confidence,
                'abstained': any(
                    a['action'] == 'ABSTAIN' 
                    for a in validator_info.get('validator_actions', [])
                )
            })
        
        # Compute metrics
        correct = sum(1 for s in hallucination_scores if s['correct'])
        accuracy = correct / len(hallucination_scores)
        
        avg_confidence = np.mean([s['validator_confidence'] for s in hallucination_scores])
        abstain_rate = sum(1 for s in hallucination_scores if s['abstained']) / len(hallucination_scores)
        
        return {
            'accuracy': accuracy,
            'avg_validator_confidence': avg_confidence,
            'abstain_rate': abstain_rate,
            'scores': hallucination_scores
        }
    
    def evaluate_reasoning(self, dataset):
        """
        Evaluate on reasoning benchmarks:
        - Sudoku-Extreme
        - Maze-Hard
        - ARC-AGI
        """
        reasoning_scores = []
        
        for problem, solution in dataset:
            output, validator_info = self.heart(
                tf.constant([self.encoder.encode(problem)]),
                training=False,
                return_validator_info=True
            )
            
            generated = self.encoder.decode(output[0].numpy())
            is_correct = generated.strip() == solution.strip()
            
            reasoning_depth = len(validator_info.get('validator_actions', []))
            
            reasoning_scores.append({
                'correct': is_correct,
                'reasoning_depth': reasoning_depth,
                'actions_taken': [a['action'] for a in validator_info.get('validator_actions', [])]
            })
        
        correct = sum(1 for s in reasoning_scores if s['correct'])
        accuracy = correct / len(reasoning_scores)
        avg_depth = np.mean([s['reasoning_depth'] for s in reasoning_scores])
        
        return {
            'accuracy': accuracy,
            'avg_reasoning_depth': avg_depth,
            'scores': reasoning_scores
        }
    
    def _check_correctness(self, generated, ground_truth):
        """Simple correctness check (would use external verifier)"""
        # Placeholder: could use entailment model, exact match, etc.
        return generated.lower().strip() == ground_truth.lower().strip()


if __name__ == "__main__":
    # Example usage
    config = HeartConfig(
        s_segments=4,
        c_cycles=2,
        r_steps=2,
        d_l=256,
        d_h=512
    )
    
    def dummy_base_model(x, past, training=False):
        batch_size = tf.shape(x)[0]
        seq_len = tf.shape(x)[1]
        return {
            'logits': tf.random.normal([batch_size, seq_len, config.n_vocab]),
            'features': tf.random.normal([batch_size, seq_len, config.n_embd])
        }
    
    trainer = HEARTTrainer(config, dummy_base_model)
    
    # Create dummy batch
    batch = {
        'input_ids': tf.constant([[1, 2, 3, 4], [5, 6, 7, 8]]),
        'target_ids': tf.constant([[2, 3, 4, 5], [6, 7, 8, 9]]),
        'alignment_labels': tf.constant([[1.0, 1.0, 0.5, 0.0], [1.0, 0.8, 0.6, 0.4]]),
        'action_labels': tf.constant([[0, 0, 1, 2], [0, 0, 0, 1]])
    }
    
    losses = trainer.train_step(batch)
    print("Training step completed!")
    print(f"Losses: {losses}")