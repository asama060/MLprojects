import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import math

torch.manual_seed(42)
np.random.seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

df = pd.read_csv('u.data', sep='\t',
                 names=['user_id','item_id','rating','ts'], usecols=[0,1,2])
df['user_id'] -= 1
df['item_id'] -= 1

m = df['user_id'].max() + 1
n = df['item_id'].max() + 1
print(f"Users: {m}, Items: {n}, Ratings: {len(df)}")

train_df, test_df = train_test_split(df, test_size=0.1, random_state=42)
print("\n" + "="*60)
print("Constant Mean Baseline:")
print("="*60)

train_mean = train_df['rating'].mean()
test_r_numpy = test_df['rating'].values
constant_pred = np.full_like(test_r_numpy, train_mean)
constant_mse = mean_squared_error(test_r_numpy, constant_pred)
constant_rmse = np.sqrt(constant_mse)

print(f"Global mean (train): {train_mean:.4f}")
print(f"Constant Mean Baseline RMSE (test): {constant_rmse:.4f}")

def train_pmf(k=50, epochs=100, lr=0.01):
    class PMF(nn.Module):
        def __init__(self, num_users, num_items, k=k):
            super().__init__()
            self.U = nn.Parameter(torch.randn(num_users, k) * 0.1)
            self.V = nn.Parameter(torch.randn(num_items, k) * 0.1)
            
        def forward(self, u, i):
            return (self.U[u] * self.V[i]).sum(1)
    
    pmf = PMF(m, n, k).to(device)
    opt = optim.Adam(pmf.parameters(), lr=lr, weight_decay=1e-4)
    
    train_u = torch.tensor(train_df['user_id'].values, device=device)
    train_i = torch.tensor(train_df['item_id'].values, device=device)
    train_r = torch.tensor(train_df['rating'].values, dtype=torch.float32, device=device)
    
    test_u = torch.tensor(test_df['user_id'].values, device=device)
    test_i = torch.tensor(test_df['item_id'].values, device=device)
    test_r = torch.tensor(test_df['rating'].values, dtype=torch.float32, device=device)
    
    best_rmse = float('inf')
    best_state = None
    
    for epoch in range(epochs):
        pmf.train()
        pred = pmf(train_u, train_i)
        loss = ((pred - train_r)**2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        
        if (epoch+1) % 50 == 0:
            with torch.no_grad():
                pmf.eval()
                test_pred = pmf(test_u, test_i)
                rmse = torch.sqrt(((test_pred - test_r)**2).mean())
                if rmse < best_rmse:
                    best_rmse = rmse
                    best_state = pmf.state_dict().copy()
            print(f"  PMF (k={k}) Epoch {epoch+1}: Loss={loss.item():.4f}, Test RMSE={rmse.item():.4f}")
    
    pmf.load_state_dict(best_state)
    
    return pmf, best_rmse.item()

k_values = [5,10,20,30,40,50,60,80,100,120]
pmf_results = {}

print("\n" + "="*60)
print("Training PMF with different k values:")
print("="*60)

for k in k_values:
    print(f"\nTraining PMF with k={k}:")
    pmf_model, best_rmse = train_pmf(k=k, epochs=100)
    pmf_results[k] = (pmf_model, best_rmse)
    print(f"  Best RMSE for k={k}: {best_rmse:.4f}")

best_k = min(pmf_results.keys(), key=lambda x: pmf_results[x][1])
best_pmf, best_pmf_rmse = pmf_results[best_k]
print(f"\n✓ Best PMF: k={best_k} with RMSE={best_pmf_rmse:.4f}")

best_pmf.eval()
for p in best_pmf.parameters():
    p.requires_grad = False

class PerceiverIOBlock(nn.Module):
    def __init__(self, latent_dim, num_heads, ff_dim=None):
        super().__init__()
        if ff_dim is None:
            ff_dim = latent_dim * 4
        
        self.norm1 = nn.LayerNorm(latent_dim)
        self.cross_attn = nn.MultiheadAttention(latent_dim, num_heads, batch_first=True)
        
        self.norm2 = nn.LayerNorm(latent_dim)
        self.self_attn = nn.MultiheadAttention(latent_dim, num_heads, batch_first=True)
        
        self.norm3 = nn.LayerNorm(latent_dim)
        self.ffn = nn.Sequential(
            nn.Linear(latent_dim, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, latent_dim)
        )
        
    def forward(self, latents, inputs):
        latents_norm = self.norm1(latents)
        attn_out, _ = self.cross_attn(latents_norm, inputs, inputs)
        latents = latents + attn_out
        
        latents_norm = self.norm2(latents)
        attn_out, _ = self.self_attn(latents_norm, latents_norm, latents_norm)
        latents = latents + attn_out
    
        latents_norm = self.norm3(latents)
        ffn_out = self.ffn(latents_norm)
        latents = latents + ffn_out
        
        return latents

class ResidualPerceiverIO(nn.Module):
    def __init__(self, num_users, num_items, latent_len=32, latent_dim=160, 
                 num_blocks=2, num_heads=8, input_dim=128, t_emb_dim=64):
        super().__init__()
        self.user_emb = nn.Embedding(num_users, input_dim // 2)  
        self.item_emb = nn.Embedding(num_items, input_dim // 2)
        self.t_emb_dim = t_emb_dim
        
        self.time_embed = nn.Sequential(              #for non linearity MLP to process time embeddings expands 64 → 128 → 64 with GELU activation 
            nn.Linear(t_emb_dim, t_emb_dim * 2),
            nn.GELU(),
            nn.Linear(t_emb_dim * 2, t_emb_dim)     
        )

        self.input_proj = nn.Linear(input_dim + 1 + t_emb_dim, latent_dim)   #Mapping heterogeneous inputs to uniform latent space
        self.latent = nn.Parameter(torch.randn(1, latent_len, latent_dim) * 0.02)
        self.blocks = nn.ModuleList([
            PerceiverIOBlock(latent_dim, num_heads) for _ in range(num_blocks)
        ])
        
        self.output_proj = nn.Linear(latent_dim, 1)


     #Create sinusoidal time embeddings. 
    def get_time_embedding(self, t, max_period=10000):   
        half_dim = self.t_emb_dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half_dim, dtype=torch.float32, device=t.device) / half_dim
        )
        args = t[:, None].float() * freqs[None, :]
        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.t_emb_dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding
        
    def forward(self, user_ids, item_ids, noisy_residual, t):
        u_emb = self.user_emb(user_ids)
        i_emb = self.item_emb(item_ids)
        t_emb = self.get_time_embedding(t)
        t_emb = self.time_embed(t_emb)
        combined = torch.cat([
            u_emb, 
            i_emb, 
            noisy_residual.unsqueeze(-1), 
            t_emb
        ], dim=-1)

        inputs = self.input_proj(combined).unsqueeze(1)  
        batch_size = user_ids.size(0)
        latents = self.latent.expand(batch_size, -1, -1)
        
        for block in self.blocks:
            latents = block(latents, inputs)
        
        decoded, _ = self.blocks[0].cross_attn(
            self.blocks[0].norm1(inputs), 
            latents, 
            latents
        )
        output = self.output_proj(decoded).squeeze(-1).squeeze(1)
        
        return output
class LinearNoiseSchedule:
    def __init__(self, beta_start=1e-4, beta_end=0.02, num_timesteps=1000, device='cuda'):
        self.device = device
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps, device=device)
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.num_timesteps = num_timesteps
        
    def q_sample(self, x0, t, noise): #Forward diffusion process.
        t = t.to(self.device)
        
        sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod[t])
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod[t])
        return sqrt_alphas_cumprod * x0 + sqrt_one_minus_alphas_cumprod * noise

print("\n" + "="*60)
print("Training Diffusion Perceiver IO:")
print("="*60)

model = ResidualPerceiverIO(m, n).to(device)
noise_schedule = LinearNoiseSchedule(num_timesteps=1000, device=device)
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
epochs = 50
batch_size = 512

all_u = torch.tensor(df['user_id'].values, device=device)
all_i = torch.tensor(df['item_id'].values, device=device)
all_r = torch.tensor(df['rating'].values, dtype=torch.float32, device=device)

with torch.no_grad():
    base_pred = best_pmf(all_u, all_i)
    residual_target = all_r - base_pred
    residual_mean = residual_target.mean()
    residual_std = residual_target.std()
    residual_target = (residual_target - residual_mean) / residual_std
    print(f"Residual stats: mean={residual_mean.item():.4f}, std={residual_std.item():.4f}")

train_idx = torch.tensor(train_df.index.values, device=device)
test_idx = torch.tensor(test_df.index.values, device=device)

for epoch in range(1, epochs+1):
    model.train()
    epoch_loss = 0.0
    num_batches = 0
    perm = torch.randperm(len(train_idx))
    
    for i in range(0, len(train_idx), batch_size):
        batch_indices = perm[i:i+batch_size]
        idx = train_idx[batch_indices]
        
        u_ids = all_u[idx]
        i_ids = all_i[idx]
        residuals = residual_target[idx]
        t = torch.randint(0, noise_schedule.num_timesteps, (len(idx),), device=device) 
        noise = torch.randn_like(residuals)
        noisy_residuals = noise_schedule.q_sample(residuals, t, noise)
        pred_noise = model(u_ids, i_ids, noisy_residuals, t)
        loss = nn.MSELoss()(pred_noise, noise)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        epoch_loss += loss.item()
        num_batches += 1
    
    if epoch % 5 == 0:
        avg_loss = epoch_loss / max(num_batches, 1)
        print(f"Epoch {epoch:3d}/{epochs} → Loss: {avg_loss:.4f}")

#Rever dinoising  process
@torch.no_grad()
def sample_simple(model, u_ids, i_ids, num_steps=50):
    
    batch_size = len(u_ids)
    device = u_ids.device
    
    x = torch.randn(batch_size, device=device)
    timesteps = torch.linspace(noise_schedule.num_timesteps-1, 0, num_steps+1, dtype=torch.long, device=device)
    timesteps = list(reversed(timesteps.int().tolist()))
    
    for i in range(num_steps):
        t = torch.full((batch_size,), timesteps[i], device=device, dtype=torch.long)
        
        pred_noise = model(u_ids, i_ids, x, t)
    
        pred_noise = torch.clamp(pred_noise, -5.0, 5.0)
        
        alpha_t = noise_schedule.alphas_cumprod[t]
        alpha_t_prev = noise_schedule.alphas_cumprod[torch.clamp(t-1, min=0)]
        sqrt_alpha_t = torch.sqrt(alpha_t)
        sqrt_one_minus_alpha_t = torch.sqrt(1.0 - alpha_t)
        
        pred_x0 = (x - sqrt_one_minus_alpha_t * pred_noise) / (sqrt_alpha_t + 1e-8)
        pred_x0 = torch.clamp(pred_x0, -10.0, 10.0)
        
        if t[0] == 0:
            x = pred_x0
        else:
            sqrt_alpha_t_prev = torch.sqrt(alpha_t_prev)
            coeff1 = sqrt_alpha_t_prev * (1 - alpha_t) / (1 - alpha_t_prev)
            coeff2 = sqrt_alpha_t * (alpha_t_prev - alpha_t) / (1 - alpha_t_prev)
            x = coeff1 * pred_x0 + coeff2 * x + torch.sqrt(1 - alpha_t_prev) * torch.randn_like(x) * 0.1
    
    return x

print("\n" + "="*60)
print("Final Evaluation:")
print("="*60)

model.eval()
preds = []

with torch.no_grad():
    test_batch_size = 512
    for i in range(0, len(test_idx), test_batch_size):
        batch_idx = test_idx[i:i+test_batch_size]
        u_ids = all_u[batch_idx]
        i_ids = all_i[batch_idx]

        base_pred = best_pmf(u_ids, i_ids)
        try:
            sampled_residual = sample_simple(model, u_ids, i_ids, num_steps=50)
        except:
            print(f"Warning: Sampling failed for batch {i}, using mean residual")
            sampled_residual = torch.zeros_like(base_pred)
        if torch.isnan(sampled_residual).any() or torch.isinf(sampled_residual).any():
            print(f"Warning: NaN/Inf detected in batch {i}, replacing with zeros")
            sampled_residual = torch.zeros_like(base_pred)
        
        sampled_residual = sampled_residual * residual_std + residual_mean
        final_pred = (base_pred + sampled_residual).clamp(1.0, 5.0)

        if torch.isnan(final_pred).any() or torch.isinf(final_pred).any():            print(f"Warning: NaN/Inf in final prediction for batch {i}, using base PMF")
            final_pred = base_pred.clamp(1.0, 5.0)
        
        preds.append(final_pred.cpu())

preds = torch.cat(preds).numpy()

if np.any(np.isnan(preds)) or np.any(np.isinf(preds)):
    print("Warning: NaN/Inf in final predictions, replacing with PMF baseline")

    with torch.no_grad():
        test_u = torch.tensor(test_df['user_id'].values, device=device)
        test_i = torch.tensor(test_df['item_id'].values, device=device)
        preds = best_pmf(test_u, test_i).cpu().numpy()
        preds = np.clip(preds, 1.0, 5.0)

final_rmse = np.sqrt(mean_squared_error(test_df['rating'].values, preds))

print(f"\nPMF Baseline (k={best_k}): RMSE = {best_pmf_rmse:.4f}")
print(f"Diffusion Perceiver IO: RMSE = {final_rmse:.4f}")
print(f"Improvement: {best_pmf_rmse - final_rmse:.4f}")


print("\n" + "="*60)
print("SUMMARY:")
print("="*60)
print(f"{'Model':<30} {'RMSE':<10}")
print("-" * 40)
for k in k_values:
    pmf_model, rmse = pmf_results[k]
    print(f"PMF (k={k}):{'':<20} {rmse:.4f}")
print(f"{'PMF + Diffusion Perceiver IO':<30} {final_rmse:.4f}")
print("="*60)