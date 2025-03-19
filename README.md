# Safe Gaussian Soft Actor-Critic Deep Reinforcement Learning Algorithm (SGDSAC)

To improve the energy efficiency of hybrid tracked tractors, we have developed a novel safe deep reinforcement learning (DRL) algorithm called **Safe Gaussian Distributed Soft Actor-Critic (SGDSAC)**. This algorithm combines our foundational DRL algorithm, **Gaussian Distributed Soft Actor-Critic (GDSAC)**, with a **Safety Monitor**.

---

## Key Features of SGDSAC

### 1. **Soft Q-Distribution Learning**
We extend the distributed Bellman operator and distributed policy evaluation introduced by Bellemare et al. [1] within the soft Q-learning framework. We provide a proof of convergence for this extension and model the soft Q distribution as a Gaussian distribution. Furthermore, we derive a projection formula for the target soft Q distribution that adheres to the temporal difference format in the soft Q-learning framework. Building on these foundations, we propose:
- A **dual soft Q-distribution learning mechanism** using inverse variance weighting.
- A **3-sigma learning rule** to improve learning efficiency and stability.

[1] Bellemare M G, Dabney W, Munos R. A distributional perspective on reinforcement learning[C]//International conference on machine learning. PMLR, 2017: 449-458.

---

### 2. **Safety Mechanisms**
To ensure safe interaction between the actor network and the environment, we introduce two categories of safety constraints:
1. **Simple Constraints**: Addressed using an **action-masking method**.
2. **Anti-Collision Constraints**: Managed through a unified **safety monitor** based on:
   - Control-affine systems,
   - Safety control using energy functions, and
   - Quadratic programming.

---

## Simulink-DLL-Python Training Framework

To facilitate the **joint training of Python and Simulink models**, we designed a **Simulink-DLL-Python training framework**.

### Key Advantages:
1. **Efficiency**:  
   Simulink models are compiled into DLLs, enabling seamless interaction with Python. DLLs offer:
   - High-speed execution
   - Minimal memory consumption

2. **Quick Setup**:  
   Detailed examples are provided in the **Simulink-DLL-Python-Demo** directory. Users can quickly set up a DRL training environment by following our framework.

3. **Real-World Applicability**:  
   Simulink enables the construction of controlled objects that closely resemble real-world systems. As a result, DRL agents trained with this framework have the potential to be directly deployed in practical control tasks.

---

## Directory Structure

```plaintext
.
├── Algorithm/                   # Source code for SGDSAC, GDSAC, SAC, and TD3 algorithms, along with their corresponding training results
├── Simulink-DLL-Python-Demo/    # Examples of Simulink-DLL-Python integration
└── README.md                    # Project documentation
