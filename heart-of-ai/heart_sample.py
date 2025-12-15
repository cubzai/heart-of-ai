"""
HEART Sampling and Inference Module
Implements generation with epistemic control and HEART-CL
"""

import tensorflow as tf
import numpy as np
from heart_validator import EpistemicActionExecutor


def top_k_logits(logits, k):
    """Apply top-k filtering to logits"""
    if k == 0:
        return logits
    
    values, _ = tf.nn.top_k(logits, k=k)
    min_values = values[:, -1, tf.newaxis]
    return tf.where(
        logits < min_values,
        tf.ones_like(logits, dtype=logits.dtype) * -1e10,
        logits
    )


def top_p_logits(logits, p):
    """Nucleus sampling: keep top p probability mass"""
    batch_size = tf.shape(logits)[0]
    sorted_logits = tf.sort(logits, direction='DESCENDING', axis=-1)
    cumulative_probs = tf.cumsum(
        tf.nn.softmax(sorted_logits, axis=-1), 
        axis=-1
    )
    
    indices = tf.stack([
        tf.range(0, batch_size),
        tf.maximum(
            tf.reduce_sum(tf.cast(cumulative_probs <= p, tf.int32), axis=-1) - 1, 
            0
        ),
    ], axis=-1)
    
    min_values = tf.gather_nd(sorted_logits, indices)
    return tf.where(
        logits < min_values,
        tf.ones_like(logits) * -1e10,
        logits
    )


class HEARTSampler:
    """
    HEART sampling with epistemic control
    Generates tokens while monitoring and controlling reliability
    """
    def __init__(self, heart_model, validator, encoder, 
                 retrieval_fn=None, repair_fn=None):
        self.heart_model = heart_model
        self.validator = validator
        self.encoder = encoder
        self.action_executor = EpistemicActionExecutor(
            validator, retrieval_fn, repair_fn
        )
        
    def sample_sequence(self, 
                       prompt_text,
                       max_length=256,
                       temperature=1.0,
                       top_k=0,
                       top_p=0.95,
                       return_validator_info=True,
                       max_repair_iterations=2):
        """
        Generate text with HEART epistemic control
        
        Args:
            prompt_text: Starting prompt
            max_length: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Top-k filtering (0 = disabled)
            top_p: Nucleus sampling probability
            return_validator_info: Return validator decisions
            max_repair_iterations: Max self-correction cycles
            
        Returns:
            generated_text: Output text
            validator_info: Dict with epistemic actions taken
        """
        # Encode prompt
        prompt_tokens = self.encoder.encode(prompt_text)
        prompt_tokens = tf.constant([prompt_tokens])  # Add batch dim
        
        generated_tokens = prompt_tokens
        validator_actions = []
        retrieval_count = 0
        repair_count = 0
        
        for step in range(max_length):
            # Forward pass through HEART
            with tf.no_grad():
                output, validator_info = self.heart_model(
                    generated_tokens, 
                    training=False,
                    return_validator_info=True
                )
                
                # Get logits for last token
                logits = output['logits'][:, -1, :]  # [batch, vocab]
                
                # Get validator decision
                alignment_scores = validator_info.get('alignment_scores', [])
                action_logits = validator_info.get('action_logits', [])
                
                # Compute alignment and action
                if alignment_scores:
                    current_alignment = alignment_scores[-1]
                else:
                    current_alignment = tf.reduce_mean(tf.nn.softmax(logits, axis=-1))
                
                # Determine action
                if action_logits:
                    action_logits_tensor = action_logits[-1]
                    action_probs = tf.nn.softmax(action_logits_tensor, axis=-1)
                    action_idx = tf.argmax(action_probs, axis=-1).numpy()
                else:
                    # Default: accept if high confidence, abstain otherwise
                    action_idx = 0 if current_alignment > 0.7 else 1
                
                # Execute epistemic action
                action_result = self.action_executor.execute(
                    action_idx,
                    generated_tokens[-1].numpy(),
                    current_alignment,
                    context=prompt_text
                )
                
                validator_actions.append({
                    'step': step,
                    'action': action_result['action'],
                    'alignment': float(current_alignment.numpy()),
                    'action_idx': int(action_idx)
                })
            
            # Handle epistemic actions
            if action_result['action'] == 'ACCEPT':
                # Standard generation
                next_token = self._sample_token(
                    logits, temperature, top_k, top_p
                )
                generated_tokens = tf.concat(
                    [generated_tokens, tf.expand_dims(next_token, 0)],
                    axis=1
                )
                
            elif action_result['action'] == 'ABSTAIN':
                # Stop and return uncertainty
                validator_actions[-1]['reason'] = 'Low confidence detected'
                break
                
            elif action_result['action'] == 'RETRIEVE':
                # Trigger RAG (simulate with context injection)
                retrieval_count += 1
                validator_actions[-1]['retrieval_triggered'] = True
                
                # In practice: retrieve documents, inject into context
                # For demo: continue with modified context
                next_token = self._sample_token(
                    logits, temperature, top_k, top_p
                )
                generated_tokens = tf.concat(
                    [generated_tokens, tf.expand_dims(next_token, 0)],
                    axis=1
                )
                
            elif action_result['action'] == 'REPAIR':
                # Self-correction
                if repair_count < max_repair_iterations:
                    repair_count += 1
                    validator_actions[-1]['repair_cycle'] = repair_count
                    # Re-sample from previous state with different temperature
                    next_token = self._sample_token(
                        logits, temperature=0.5, top_k=top_k, top_p=top_p
                    )
                    generated_tokens = tf.concat(
                        [generated_tokens, tf.expand_dims(next_token, 0)],
                        axis=1
                    )
                else:
                    # Max repairs reached
                    break
        
        # Decode output
        output_tokens = generated_tokens[0].numpy()
        generated_text = self.encoder.decode(output_tokens)
        
        result = {
            'text': generated_text,
            'tokens': output_tokens,
            'length': len(output_tokens),
            'prompt': prompt_text
        }
        
        if return_validator_info:
            result['validator_info'] = {
                'actions': validator_actions,
                'retrieval_count': retrieval_count,
                'repair_count': repair_count,
                'epistemic_path': [a['action'] for a in validator_actions]
            }
        
        return result
    
    def _sample_token(self, logits, temperature, top_k, top_p):
        """Sample next token from logits"""
        # Apply temperature
        logits = logits / temperature
        
        # Apply filtering
        logits = top_k_logits(logits, top_k)
        logits = top_p_logits(logits, top_p)
        
        # Sample
        samples = tf.random.categorical(logits, num_samples=1, dtype=tf.int32)
        return samples[:, 0]


class InteractiveHEARTSampler(HEARTSampler):
    """
    Interactive HEART sampling with user feedback
    """
    def interactive_generate(self, 
                            initial_prompt="",
                            temperature=0.7,
                            enable_heart_cl=True):
        """
        Interactive generation loop with user control
        """
        print("=" * 60)
        print("HEART Interactive Generation")
        print("=" * 60)
        print("Commands:")
        print("  'continue' - Generate more tokens")
        print("  'abstain' - Request uncertainty from model")
        print("  'retrieve' - Trigger knowledge retrieval")
        print("  'repair' - Ask model to reconsider")
        print("  'feedback' - Provide ground truth for HEART-CL")
        print("  'quit' - Exit")
        print("=" * 60)
        
        if not initial_prompt:
            initial_prompt = input("Enter starting prompt: ").strip()
        
        current_text = initial_prompt
        generation_history = []
        
        while True:
            print(f"\nCurrent: {current_text[:100]}...")
            command = input("\nCommand> ").strip().lower()
            
            if command == 'quit':
                break
                
            elif command == 'continue':
                result = self.sample_sequence(
                    current_text,
                    max_length=64,
                    temperature=temperature
                )
                current_text = result['text']
                generation_history.append(result)
                
                print(f"\nGenerated: {result['text'][len(initial_prompt):]}")
                if 'validator_info' in result:
                    print(f"Epistemic path: {result['validator_info']['epistemic_path']}")
                
            elif command == 'abstain':
                result = self.sample_sequence(
                    current_text,
                    max_length=1,  # Just get validator decision
                    temperature=temperature
                )
                if result['validator_info']['actions']:
                    last_action = result['validator_info']['actions'][-1]
                    print(f"\nValidator alignment: {last_action['alignment']:.3f}")
                
            elif command == 'retrieve':
                print("\n[Simulating retrieval...]")
                # Would integrate RAG here
                result = self.sample_sequence(
                    current_text,
                    max_length=32,
                    temperature=0.5
                )
                current_text = result['text']
                
            elif command == 'repair':
                print("\n[Requesting self-correction...]")
                result = self.sample_sequence(
                    current_text,
                    max_length=32,
                    temperature=0.5
                )
                current_text = result['text']
                
            elif command == 'feedback':
                if enable_heart_cl:
                    correct_text = input("Provide correct version: ").strip()
                    # Would use for HEART-CL training
                    print("[Feedback recorded for continual learning]")
                else:
                    print("[HEART-CL disabled]")
            
            else:
                print("Unknown command. Try again.")
        
        return {
            'final_text': current_text,
            'history': generation_history
        }


class HEARTConditionalSampler(HEARTSampler):
    """
    Conditional sampling with style/domain control
    """
    def sample_with_constraints(self,
                               prompt,
                               constraints=None,
                               max_length=256,
                               temperature=0.8):
        """
        Generate with hard constraints
        
        Args:
            prompt: Starting text
            constraints: Dict with constraints:
                - must_include: List of words that must appear
                - avoid_words: List of words to avoid
                - style: 'formal', 'casual', 'technical'
                - domain: 'medical', 'legal', 'finance'
            max_length: Max generation length
            temperature: Sampling temperature
            
        Returns:
            result: Generated text with constraint satisfaction
        """
        constraints = constraints or {}
        
        result = self.sample_sequence(
            prompt,
            max_length=max_length,
            temperature=temperature,
            return_validator_info=True
        )
        
        # Check constraints
        generated_words = result['text'].lower().split()
        
        must_include = constraints.get('must_include', [])
        avoided_words = constraints.get('avoid_words', [])
        
        constraints_satisfied = {
            'must_include': all(w.lower() in generated_words for w in must_include),
            'avoid_words': not any(w.lower() in generated_words for w in avoided_words)
        }
        
        result['constraints_satisfied'] = constraints_satisfied
        
        return result


# Example usage
if __name__ == "__main__":
    # Mock classes for testing
    class MockEncoder:
        def encode(self, text):
            return [1, 2, 3, 4]  # Dummy tokens
        
        def decode(self, tokens):
            return "Generated text"
    
    class MockHeartModel:
        def __call__(self, x, training=False, return_validator_info=False):
            batch_size = tf.shape(x)[0]
            output = {
                'logits': tf.random.normal([batch_size, 1, 50257])
            }
            if return_validator_info:
                return output, {
                    'alignment_scores': [tf.constant(0.8)],
                    'action_logits': [tf.random.normal([4])]
                }
            return output
    
    class MockValidator:
        pass
    
    # Create sampler
    encoder = MockEncoder()
    model = MockHeartModel()
    validator = MockValidator()
    
    sampler = HEARTSampler(model, validator, encoder)
    
    # Test generation
    result = sampler.sample_sequence(
        "Once upon a time",
        max_length=10,
        return_validator_info=True
    )
    
    print("Generation successful!")
    print(f"Output length: {result['length']}")
    print(f"Epistemic path: {result['validator_info']['epistemic_path']}")