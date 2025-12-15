"""
HEART - Complete Integration Module
Hierarchical Epistemic Architecture for Reasoning and Truth-seeking

This is the main entry point that combines all HEART components
"""

import tensorflow as tf
import numpy as np
from heart_model import HEART, HeartConfig, create_heart_model
from heart_validator import ValidatorNetwork, EpistemicActionExecutor
from heart_sample import HEARTSampler, InteractiveHEARTSampler
from heart_train import HEARTTrainer, HeartCLTrainer


class HeartOfAI:
    """
    Complete HEART system integrating:
    - Base language model (GPT-2)
    - L-Module for fast reasoning
    - H-Module for semantic context
    - Epistemic Validator for control
    - HEART-CL for continual learning
    """
    
    def __init__(self, 
                 base_model_path='models/gpt2',
                 config_dict=None,
                 mode='inference'):
        """
        Initialize HEART system
        
        Args:
            base_model_path: Path to base GPT-2 model
            config_dict: Configuration overrides
            mode: 'inference', 'training', or 'interactive'
        """
        # Load or create configuration
        if config_dict is None:
            config_dict = {}
        
        self.config = HeartConfig(
            s_segments=config_dict.get('s_segments', 4),
            c_cycles=config_dict.get('c_cycles', 2),
            r_steps=config_dict.get('r_steps', 2),
            d_l=config_dict.get('d_l', 256),
            d_h=config_dict.get('d_h', 512),
            enable_heart_cl=config_dict.get('enable_heart_cl', True),
            learning_rate=config_dict.get('learning_rate', 5e-3),
            alignment_weight=config_dict.get('alignment_weight', 0.5),
            retrieve_threshold=config_dict.get('retrieve_threshold', 0.6)
        )
        
        self.mode = mode
        self.base_model_path = base_model_path
        
        # Initialize components
        self._init_components()
        
    def _init_components(self):
        """Initialize all HEART components"""
        # Base LM function (would load actual GPT-2)
        self.base_model_fn = self._create_base_model_fn()
        
        # Create HEART model
        self.heart_model = create_heart_model(self.config, self.base_model_fn)
        
        # Create validator network
        self.validator_network = ValidatorNetwork(
            hidden_dim=512, 
            n_actions=4
        )
        
        # Create action executor
        self.action_executor = EpistemicActionExecutor(
            self.validator_network
        )
        
        # Create trainer if in training mode
        if self.mode == 'training':
            self.trainer = HEARTTrainer(self.config, self.base_model_fn)
        
        # Create HEART-CL if enabled
        if self.config.enable_heart_cl:
            self.heart_cl = HeartCLTrainer(self.heart_model, self.config)
        
        print("✓ HEART system initialized successfully")
        print(f"  Configuration:")
        print(f"    Supervision segments (S): {self.config.s_segments}")
        print(f"    Cycles per segment (C): {self.config.c_cycles}")
        print(f"    Steps per cycle (R): {self.config.r_steps}")
        print(f"    Effective depth: ~{self.config.d_eff} layers")
        print(f"    L-module dim: {self.config.d_l}")
        print(f"    H-module dim: {self.config.d_h}")
        print(f"    HEART-CL enabled: {self.config.enable_heart_cl}")
    
    def _create_base_model_fn(self):
        """Create base language model function (GPT-2 wrapper)"""
        def base_model_fn(x, past=None, training=False):
            # In real implementation, would load GPT-2 from checkpoint
            # For now, return dummy features
            batch_size = tf.shape(x)[0]
            seq_len = tf.shape(x)[1]
            return {
                'logits': tf.random.normal([batch_size, seq_len, self.config.n_vocab]),
                'features': tf.random.normal([batch_size, seq_len, self.config.n_embd])
            }
        return base_model_fn
    
    # =================== INFERENCE API ===================
    
    def generate(self,
                prompt,
                max_length=256,
                temperature=0.7,
                top_k=40,
                top_p=0.95,
                return_epistemic_info=True):
        """
        Generate text with HEART epistemic control
        
        Args:
            prompt: Starting text
            max_length: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Top-k filtering (0 = disabled)
            top_p: Nucleus sampling probability
            return_epistemic_info: Return validator decisions
            
        Returns:
            dict with 'text', 'tokens', and optionally 'epistemic_info'
        """
        # Encode prompt (in practice would use real encoder)
        prompt_tokens = [1, 2, 3]  # Dummy tokens
        
        # Generate with HEART
        output, validator_info = self.heart_model(
            tf.constant([prompt_tokens]),
            training=False,
            return_validator_info=True
        )
        
        result = {
            'text': f"{prompt} [generated text]",  # Would decode properly
            'tokens': output[0].numpy(),
            'prompt': prompt,
            'length': len(output[0])
        }
        
        if return_epistemic_info:
            result['epistemic_info'] = {
                'actions': validator_info.get('validator_actions', []),
                'alignment_scores': validator_info.get('alignment_scores', []),
                'reasoning_depth': len(validator_info.get('validator_actions', []))
            }
        
        return result
    
    def interactive_mode(self, initial_prompt=""):
        """
        Interactive generation with user control
        """
        # Note: Would need proper encoder for this
        print("\n" + "="*70)
        print("HEART Interactive Generation Mode")
        print("="*70)
        print("Supported commands:")
        print("  'gen'       - Generate next tokens")
        print("  'abstain'   - Request model confidence")
        print("  'retrieve'  - Trigger knowledge retrieval")
        print("  'repair'    - Request self-correction")
        print("  'explain'   - Get explanation of validator decision")
        print("  'help'      - Show this menu")
        print("  'exit'      - Exit interactive mode")
        print("="*70)
        
        current_text = initial_prompt or input("\nEnter starting prompt: ").strip()
        action_history = []
        
        while True:
            print(f"\nCurrent text: {current_text[:80]}...")
            cmd = input("Command > ").strip().lower()
            
            if cmd == 'exit':
                print("Exiting interactive mode...")
                break
            
            elif cmd == 'gen':
                result = self.generate(current_text, max_length=50)
                current_text = result['text']
                action_history.extend(result['epistemic_info']['actions'])
                print(f"Generated: {current_text}")
            
            elif cmd == 'abstain':
                print("[Checking model confidence...]")
                # Would call validator
                print("Model confidence: [retrieving...]")
            
            elif cmd == 'retrieve':
                print("[Triggering knowledge retrieval...]")
                # Would integrate RAG
                print("Retrieved: [simulated context]")
            
            elif cmd == 'repair':
                print("[Requesting self-correction...]")
                # Would run repair cycle
                current_text = current_text + " [corrected]"
            
            elif cmd == 'explain':
                print("[Explanation of validator decision:]")
                print("- Alignment score: 0.85")
                print("- Action: ACCEPT (generate)")
                print("- Key features: [concept1, concept2]")
            
            elif cmd == 'help':
                print("Available commands: gen, abstain, retrieve, repair, explain, exit")
            
            else:
                print(f"Unknown command: {cmd}")
        
        return {'final_text': current_text, 'action_history': action_history}
    
    # =================== TRAINING API ===================
    
    def train(self,
             train_dataset,
             val_dataset=None,
             epochs=3,
             steps_per_epoch=None,
             checkpoint_dir='checkpoints/heart'):
        """
        Train HEART model
        
        Args:
            train_dataset: Training data
            val_dataset: Validation data
            epochs: Number of epochs
            steps_per_epoch: Steps per epoch (None = all)
            checkpoint_dir: Where to save checkpoints
        """
        if self.mode != 'training':
            print("Warning: Model not in training mode. Switching...")
            self.mode = 'training'
            self._init_components()
        
        print("\n" + "="*70)
        print("Starting HEART Training")
        print("="*70)
        print(f"Total epochs: {epochs}")
        print(f"Steps per epoch: {steps_per_epoch or 'all'}")
        print(f"Checkpoint directory: {checkpoint_dir}")
        print("="*70)
        
        for epoch in range(epochs):
            print(f"\n--- Epoch {epoch + 1}/{epochs} ---")
            
            # Train
            epoch_losses = self.trainer.train_epoch(
                train_dataset, 
                steps_per_epoch=steps_per_epoch
            )
            
            print(f"Training losses: {epoch_losses}")
            
            # Validate
            if val_dataset:
                val_loss = self.trainer.validate(val_dataset)
                print(f"Validation loss: {val_loss:.4f}")
            
            # Save checkpoint
            # self.heart_model.save_weights(f"{checkpoint_dir}/epoch_{epoch+1}")
        
        print("\n✓ Training completed!")
        return epoch_losses
    
    # =================== HEART-CL API ===================
    
    def adapt_on_feedback(self, generated_tokens, correct_tokens, 
                         learning_rate=1e-6):
        """
        Continual learning: Adapt HEART on user corrections
        
        Args:
            generated_tokens: What HEART generated
            correct_tokens: Ground truth
            learning_rate: Adaptation learning rate
            
        Returns:
            Adaptation metrics
        """
        if not self.config.enable_heart_cl:
            print("HEART-CL is disabled in configuration")
            return None
        
        adaptation_info = self.heart_cl.adapt_on_feedback(
            generated_tokens,
            correct_tokens,
            epistemic_signals={},
            learning_rate=learning_rate
        )
        
        print(f"✓ Adapted on feedback")
        print(f"  Repair loss: {adaptation_info['repair_loss']:.4f}")
        print(f"  Total adaptation loss: {adaptation_info['total_adaptation_loss']:.4f}")
        
        return adaptation_info
    
    # =================== EVALUATION API ===================
    
    def evaluate_on_benchmark(self, benchmark_name, dataset):
        """
        Evaluate HEART on standard benchmarks
        
        Supported benchmarks:
        - hallucination: TruthfulQA, HaluEval, CRAG
        - reasoning: Sudoku, Maze, ARC-AGI
        - safety: Clinical, Legal domains
        """
        from heart_train import HEARTBenchmark
        
        benchmark = HEARTBenchmark(
            self.heart_model,
            self.validator_network,
            encoder=None  # Would pass actual encoder
        )
        
        if benchmark_name == 'hallucination':
            results = benchmark.evaluate_hallucination(dataset)
        elif benchmark_name == 'reasoning':
            results = benchmark.evaluate_reasoning(dataset)
        else:
            raise ValueError(f"Unknown benchmark: {benchmark_name}")
        
        print(f"\n{'='*70}")
        print(f"Benchmark: {benchmark_name}")
        print(f"{'='*70}")
        for key, value in results.items():
            if isinstance(value, float):
                print(f"{key}: {value:.4f}")
            else:
                print(f"{key}: {value}")
        
        return results
    
    # =================== CONFIGURATION API ===================
    
    def get_config(self):
        """Get current configuration"""
        return {
            's_segments': self.config.s_segments,
            'c_cycles': self.config.c_cycles,
            'r_steps': self.config.r_steps,
            'd_l': self.config.d_l,
            'd_h': self.config.d_h,
            'd_eff': self.config.d_eff,
            'validator_hidden': self.config.validator_hidden,
            'enable_heart_cl': self.config.enable_heart_cl,
            'learning_rate': self.config.learning_rate,
            'alignment_weight': self.config.alignment_weight
        }
    
    def update_config(self, config_dict):
        """Update configuration (before training)"""
        if self.mode == 'training':
            print("Cannot update config during training. Please restart.")
            return False
        
        for key, value in config_dict.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
        
        # Reinitialize with new config
        self._init_components()
        print("✓ Configuration updated and components reinitialized")
        return True
    
    def get_status(self):
        """Get current system status"""
        return {
            'mode': self.mode,
            'model_initialized': self.heart_model is not None,
            'validator_initialized': self.validator_network is not None,
            'heart_cl_enabled': self.config.enable_heart_cl,
            'config': self.get_config()
        }


# =================== EXAMPLE USAGE ===================

def example_inference():
    """Example: Generate with epistemic control"""
    print("ALHAMDULILLAH - HEART Example: Inference")
    print("=" * 70)
    
    heart = HeartOfAI(mode='inference')
    
    result = heart.generate(
        prompt="Once upon a time",
        max_length=100,
        temperature=0.7,
        return_epistemic_info=True
    )
    
    print(f"Prompt: {result['prompt']}")
    print(f"Generated: {result['text']}")
    print(f"Epistemic path: {[a['action'] for a in result['epistemic_info']['actions']]}")
    print(f"Reasoning depth: {result['epistemic_info']['reasoning_depth']}")


def example_interactive():
    """Example: Interactive generation"""
    print("ALHAMDULILLAH - HEART Example: Interactive Mode")
    print("=" * 70)
    
    heart = HeartOfAI(mode='inference')
    heart.interactive_mode(initial_prompt="The future of AI is")


def example_training():
    """Example: Training HEART"""
    print("ALHAMDULILLAH - HEART Example: Training")
    print("=" * 70)
    
    heart = HeartOfAI(mode='training')
    
    # Create dummy dataset
    dummy_dataset = []
    
    # Train
    heart.train(
        train_dataset=dummy_dataset,
        epochs=1,
        steps_per_epoch=10
    )


def example_continual_learning():
    """Example: HEART-CL adaptation"""
    print("ALHAMDULILLAH - HEART Example: Continual Learning")
    print("=" * 70)
    
    heart = HeartOfAI(mode='inference')
    
    # Simulate feedback
    generated = tf.constant([1, 2, 3, 4, 5])
    correct = tf.constant([1, 2, 3, 4, 6])  # Last token different
    
    heart.adapt_on_feedback(generated, correct)


if __name__ == "__main__":
    print("="*70)
    print("HEART - Hierarchical Epistemic Architecture")
    print("Reasoning and Truth-seeking")
    print("By CubzAI")
    print("="*70)
    
    # Show system status
    heart = HeartOfAI(mode='inference')
    status = heart.get_status()
    
    print("\nSystem Status:")
    for key, value in status.items():
        print(f"  {key}: {value}")
    
    print("\nExample usage available:")
    print("  - example_inference()")
    print("  - example_interactive()")
    print("  - example_training()")
    print("  - example_continual_learning()")
    print("\nRun any example function to get started!")
