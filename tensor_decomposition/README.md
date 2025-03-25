
## 🎯 Goal
To demonstrate how **tensor decomposition** can extract interpretable, latent patterns from multivariate time-series data.

### 🧪 Dataset
A synthetic 3D tensor simulating measurements from:
- 4 sensors
- Over 48 time steps
- 6 distinct features

Anomalies were injected at known time-sensor-feature points to validate interpretability.

### 🧠 Method
We applied **CP (CANDECOMP/PARAFAC) decomposition** with rank=3 using Tensorly. This decomposes the 3D tensor into three matrices:
- **Time factor**: how temporal patterns align with components
- **Sensor factor**: how sensors contribute to each component
- **Feature factor**: how features load on the components

### 📊 Results
- Factor matrices reveal interpretable patterns across sensors and features.
- Clear temporal patterns are visible in the time factor.
- Spikes in component intensity can hint at structural anomalies.

### 🔍 Applications
- Time-series anomaly detection
- Feature compression
- Sensor correlation analysis
- Model input for downstream ML

### 🔗 Future Work
- Use real-world datasets (e.g., environmental, industrial, or finance)
- Apply sparse or regularized tensor models
- Integrate anomaly detection thresholds using reconstruction error

---
