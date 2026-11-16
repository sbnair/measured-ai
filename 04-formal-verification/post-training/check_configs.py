from trl import SFTConfig, DPOConfig, GRPOConfig
sft = SFTConfig(output_dir="sft-out", max_length=4096, assistant_only_loss=True,
                learning_rate=5e-6, num_train_epochs=2, bf16=False)
dpo = DPOConfig(output_dir="dpo-out", beta=0.1, loss_type=["sigmoid"], learning_rate=5e-7,
                bf16=False)
grpo = GRPOConfig(output_dir="grpo-out", num_generations=8, max_completion_length=1024,
                  beta=0.0, learning_rate=1e-6, scale_rewards="group", loss_type="dapo", bf16=False)
print("configs OK", sft.max_length, dpo.beta, grpo.num_generations)
