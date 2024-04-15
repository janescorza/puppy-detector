## Unleash Your Inner Dog Detective 🐶🕵️‍♂️ with This Deep Learning Puppy Detector!✨🐾

This repository houses the code for a **puppy detector** I have built entirely from scratch using **Python and my Deep Learning skills**. This simple project aims to **distinguish between images of genuine puppies and imposters** (other animals or objects) using a neural network model.

🤔 **Why Build from Scratch?** 🤔

This project goes beyond using pre-built libraries like TensorFlow. Instead, it focuses on understanding the core principles of neural networks. Building it from scratch allowed me to:

* 💡**Demonstrate My Chore Understanding**💡: This project demonstrates my understanding of neural network implementation, showcasing my learning journey.
* ⚙️**Grasp the Intricate Workings of Neural Networks**⚙️: Understanding the "how" and "why" behind neural networks empowers me to optimize them effectively.
* 🔨**Experiment and Refine**🔨: Understanding the core mechanics enables me to explore diverse optimization techniques and fine-tune the model for superior performance.

✨Key Optimizations✨:

* **He initialization** for weights : This sets weights with a standard deviation that's more suitable for ReLU activations, helping avoid "dying neurons" and improving convergence speed.

* **Adam optimizer** for efficient updates : Adam combines the benefits of momentum and adaptive learning rates for faster and smoother convergence, often outperforming basic gradient descent.

* **Mini-batch gradient descent** (Breaking data into smaller chunks): This enables faster updates and often improves generalization, as the model sees a more diverse set of examples in each iteration.

Ready to unleash the power of Deep Learning for adorable purposes? Check out the code, and let's discuss it while diving into the future of AI for doggo classification! 🐶💻🚀

**Getting Started:**

1. **Clone the repository:**
   ```bash
   git clone https://github.com/janescorza/puppy-detector.git
   ```
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the script:**
   ```bash
   python3 main.py
   ```

**Note on Overfitting/Underfitting:**
The model currently performs poorly for this usecase, basically due to not using a convolutional network, to what essentially is an image classification problem. 
The point of this repository is not to provide excellent classification of cats and dogs, but rather show how a neural network can be built from the chore and upwards. 
Potential improvements include: making the neural network deeper to better catch more detailed features and then enlarging the training dataset, augmenting data. 
These enhancements are pending due to current limitations with my computational resources. 
However, I am open to discussing how this repository could be fixed and improved and trying to run a deeper version of it with more resources.
