# Safe-gaussian-soft-actor-critic-deep-reinforcement-learning-algorithm

To improve the energy efficiency of hybrid tracked tractors, we have developed a novel safe deep reinforcement learning (DRL) algorithm called **Safe Gaussian Distributed Soft Actor-Critic (SGDSAC)**. This algorithm combines our foundational DRL algorithm, **Gaussian Distributed Soft Actor-Critic (GDSAC)**, with a safety supervision mechanism.

The source code for SGDSAC is available in the **Algorithm** directory.

---

## Key Features of SGDSAC

### 1. **Dual Soft Q-Distribution Learning**
We derived a projection formula for the variance of the soft Q-distribution under the soft Q-distribution learning framework, ensuring it conforms to the **temporal difference learning format**. Building upon this, we proposed:
- A **dual soft Q-distribution learning mechanism** based on inverse variance weighting.
- A **3-sigma learning rule** to enhance learning efficiency and stability.

---

### 2. **Safety Mechanisms**
To ensure safe interaction between the actor network and the environment, we introduced two categories of safety constraints:
1. **Simple Constraints**: Addressed using an **action-masking method**.
2. **State Anti-Collision Constraints**: Managed through a unified safety supervision mechanism based on:
   - Control-affine systems,
   - Safety index functions, and
   - Quadratic programming.

---

## Simulink-DLL-Python Training Framework

To further facilitate the **joint training of Python and Simulink models**, we designed a **Simulink-DLL-Python training framework**. 

### Key Advantages:
1. **Efficiency**:  
   Simulink models are compiled into DLLs, which seamlessly interact with Python. DLLs offer:
   - High speed
   - Minimal memory consumption

2. **Improved Performance**:  
   The **DLL-Python training approach** is significantly more effective than conventional Simulink-Python approaches.

3. **Quick Setup**:  
   Detailed examples are provided in the **Simulink-DLL-Python-Demo** directory. By following our framework, users can quickly set up a DRL training environment.

4. **Real-World Applicability**:  
   Using Simulink, controlled objects that closely resemble real-world systems can be constructed. As a result, DRL agents trained with this framework have the potential to be directly deployed in **practical control tasks**.

---

## Directory Structure

```plaintext
.
├── Algorithm/                   # Source code for the SGDSAC algorithm
├── Simulink-DLL-Python-Demo/    # Examples of Simulink-DLL-Python integration
└── README.md                    # Project documentation
