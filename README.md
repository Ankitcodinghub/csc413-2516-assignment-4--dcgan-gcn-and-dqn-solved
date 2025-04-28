# csc413-2516-assignment-4--dcgan-gcn-and-dqn-solved
**TO GET THIS SOLUTION VISIT:** [CSC413-2516 Assignment 4- DCGAN, GCN, and DQN Solved](https://www.ankitcodinghub.com/product/csc413-2516-programming-assignment-4-dcgan-gcn-and-dqn-solved/)


---

📩 **If you need this solution or have special requests:** **Email:** ankitcoding@gmail.com  
📱 **WhatsApp:** +1 419 877 7882  
📄 **Get a quote instantly using this form:** [Ask Homework Questions](https://www.ankitcodinghub.com/services/ask-homework-questions/)

*We deliver fast, professional, and affordable academic help.*

---

<h2>Description</h2>



<div class="kk-star-ratings kksr-auto kksr-align-center kksr-valign-top" data-payload="{&quot;align&quot;:&quot;center&quot;,&quot;id&quot;:&quot;118071&quot;,&quot;slug&quot;:&quot;default&quot;,&quot;valign&quot;:&quot;top&quot;,&quot;ignore&quot;:&quot;&quot;,&quot;reference&quot;:&quot;auto&quot;,&quot;class&quot;:&quot;&quot;,&quot;count&quot;:&quot;2&quot;,&quot;legendonly&quot;:&quot;&quot;,&quot;readonly&quot;:&quot;&quot;,&quot;score&quot;:&quot;5&quot;,&quot;starsonly&quot;:&quot;&quot;,&quot;best&quot;:&quot;5&quot;,&quot;gap&quot;:&quot;4&quot;,&quot;greet&quot;:&quot;Rate this product&quot;,&quot;legend&quot;:&quot;5\/5 - (2 votes)&quot;,&quot;size&quot;:&quot;24&quot;,&quot;title&quot;:&quot;CSC413-2516  Assignment 4- DCGAN, GCN, and DQN Solved&quot;,&quot;width&quot;:&quot;138&quot;,&quot;_legend&quot;:&quot;{score}\/{best} - ({count} {votes})&quot;,&quot;font_factor&quot;:&quot;1.25&quot;}">

<div class="kksr-stars">

<div class="kksr-stars-inactive">
            <div class="kksr-star" data-star="1" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="2" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="3" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="4" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="5" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
    </div>

<div class="kksr-stars-active" style="width: 138px;">
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
    </div>
</div>


<div class="kksr-legend" style="font-size: 19.2px;">
            5/5 - (2 votes)    </div>
    </div>
Version: 1.0

Submission: You must submit 4 files through MarkUs : a PDF file containing your writeup, titled a4-writeup.pdf, and your code files a4-dcgan.ipynb, a4-gcn.ipynb, a4-dqn.ipynb. Your writeup must be typed.

The programming assignments are individual work. See the Course Information handout for detailed policies.

Introduction

In this assignment, you’ll get hands-on experience coding and training GANs, GCN (Graph Convolution Network) as well as DQN (Deep Q-learning Network), one of Reinforcement Learning methods. This assignment is divided into three parts: in the first part, we will implement a specific type of GAN designed to process images, called a Deep Convolutional GAN (DCGAN). We’ll train the DCGAN to generate emojis from samples of random noise. In the second part, you will learn how to implement the vanilla version of GCN and GAT. In the third part, we will implement and train a DQN agent to learn how to play the CartPole balancing game. It will be fun to see your model performs much better than you on the simple game :).

Part 1: Deep Convolutional GAN (DCGAN) [4pt]

For the first part of this assignment, we will implement a Deep Convolutional GAN (DCGAN). A DCGAN is simply a GAN that uses a convolutional neural network as the discriminator, and a network composed of transposed convolutions as the generator. To implement the DCGAN, we need to specify three things: 1) the generator, 2) the discriminator, and 3) the training procedure.

We will go over each of these three components in the following subsections.

Open [DCGAN notebook link] on Colab and answer the following questions.

DCGAN

The discriminator in this DCGAN is a convolutional neural network that has the following architecture:

The DCDiscriminator class is implemented for you. We strongly recommend you to carefully read the code, in particular the __init__ method. The three stages of the generator architectures are implemented using conv and upconv functions respectively, all of which provided in Helper Modules.

Discriminator

Generator

Now, we will implement the generator of the DCGAN, which consists of a sequence of transpose convolutional layers that progressively upsample the input noise sample to generate a fake image. The generator has the following architecture:

Generator

1. [1pt] Implementation: Implement this architecture by filling in the __init__ method of the DCGenerator class, shown below. Note that the forward pass of DCGenerator is already provided for you.

Note: The original DCGAN generator uses deconv function to expand the spatial dimension. Odena et al. later found the deconv creates checker board artifacts in the generated samples. In this assignment, we will use upconv that consists of an upsampling layer followed by conv2D to replace the deconv module (analogous to the conv function used for the discriminator above) in your generator implementation.

Training Loop

Algorithm 1 Regular GAN Training Loop Pseudocode

1: procedure TrainGAN

2: Draw m training examples {x(1),…,x(m)} from the data distribution pdata

3: Draw m noise samples {z(1),…,z(m)} from the noise distribution pz

4: Generate fake images from the noise: G(z(i)) for i ∈ {1,….m}

5: Compute the discriminator loss (negative log likelihood):

6: Update the parameters of the discriminator

7: Draw m new noise samples {z(1),…,z(m)} from the noise distribution pz

8: Generate fake images from the noise: G(z(i)) for i ∈ {1,….m}

9: Compute the generator loss (negative log likelihood):

10: Update the parameters of the generator

1. [1pt] Implementation: Fill in the gan_training_loop_regular function in the GAN section of the notebook.

There are 5 numbered bullets in the code to fill in for the discriminator and 3 bullets for the generator. Note that in the original GAN paper, we want to train the Generator by minimizing log in an effort to generate better fakes. However, this was shown by Goodfellow to not provide sufficient gradients, especially early in the learning process. As a fix, we instead wish to maximize log , which is also refered as the non-saturating GAN Loss.

Experiment

1. [1pt] We will train a DCGAN to generate Windows (or Apple) emojis in the Training – GAN section of the notebook. By default, the script runs for 20000 iterations, and should take approximately 20 minutes on Colab. The script saves the output of the generator for a fixed noise sample every 200 iterations throughout training; this allows you to see how the generator improves over time. How does the generator performance evolve over time? Include in your write-up some representative samples (e.g. one early in the training, one with satisfactory image quality, and one towards the end of training, and give the iteration number for those samples. Briefly comment on the quality of the samples.

(1)

(2)

After the implementation, try turn on the least_squares_gan flag in the args_dict and train the model again. Are you able to stabilize the training? Briefly explain in 1∼2 sentences why the least squares GAN can help. You are welcome to check out the related blog posts for LSGAN.

Part 2: Graph Convolution Networks

Basics of GCN:

1. the input features of each node, xi ∈ RF (in matrix form: X ∈ R|V |×F)

2. some information about the graph structure, typically the adjacency matrix A

Each convolutional layer can be written as H(l+1) = f(H(l),A), for some function f(). The f() we are using for this assignment is in the form of f(H(l),A) = σ(Dˆ−1/2AˆDˆ−1/2H(l)W(l)), where Aˆ = A + Identity and Dˆ is diagonal node degree matrix (Dˆ−1Aˆ normalizes Aˆ such that all rows sum to one). Let A˜ = Dˆ−1/2AˆDˆ−1/2. The GCN we will implement takes two convolution layers, Z = f(X,A) = softmax(A˜ · Dropout(ReLU(AXW˜ (0))) · W(1))

Basics of GAT:

Graph Attention Network (GAT) is a novel convolution-style neural network. It operates on graphstructured data and leverages masked self-attentional layers. In this assignment, we will implement the graph attention layer.

Dataset:

The dataset we used for this assignment is Cora Sen et al. [2008]. Cora is one of standard citation network benchmark dataset (just like MNIST dataset for computer vision tasks). It that consists of 2708 scientific publications and 5429 links. Each publication is classified into one of 7 classes. Each publication is described by a word vector (length 1433) that indicates the absence/presence of the corresponding word. This is used as the features of each node for our experiment. The task is to perform node classification (predict which class each node belongs to).

Experiments:

Open [GCN notebook link] on Colab and answer the following questions.

1. [1pt] Implementation of Graph Convolution Layer Complete the code for GraphConvolution() Class

2. [1pt] Implementation of Graph Convolution Network Complete the code for GCN() Class

3. [0.5pt] Train your Graph Convolution Network

After implementing the required classes, now you can train your GCN. You can play with the hyperparameters in args.

4. [2pt] Implementation of Graph Attention Layer Complete the code for GraphAttentionLayer() Class

5. [0.5pt] Train your Graph Convolution Network

After implementing the required classes, now you can train your GAT. You can play with the hyperparameters in args.

6. [0.5pt] Compare your models

Compare the evaluation results for Vanilla GCN and GAT. Comment on the discrepancy in their performance (if any) and briefly explain why you think it’s the case (in 1-2 sentences).

Part 3: Deep Q-Learning Network (DQN) [4pt]

DQN Overview

Reinforcement learning defines an environment for the agent to perform certain actions (according to the policy) that maximize the reward at every time stamp. Essentially, our aim is to train a agent that tries to maximize the discounted, cumulative reward . Because we assume there can be infinite time stamps, the discount factor, γ, is a constant between 0 and 1 that ensures the sum converges. It makes rewards from the uncertain far future less important for our agent than the ones in the near future.

The idea of Q-learning is that if we have a function Q∗(state,action) that outputs the maximum expected cumulative reward achievable from a given state-action pair, we could easily construct a policy (action selection rule) that maximizes the reward:

π∗(s) = argmax Q∗(s,a) (3) a

However, we don’t know everything about the world, so we don’t have access to Q∗. But, since neural networks are universal function approximators, we can simply create one and train it to resemble Q∗. For our training update rule, we will use a fact that every Q function for some policies obeys the Bellman equation:

Qπ(s,a) = r + γQπ(s′,π(s′)) (4)

An intuitive explanation of the structure of the Bellman equation is as follows. Suppose that the agent has received reward rt at the current state, then the maximum discounted reward from this point onward is equal to the current reward plus the maximum expected discounted reward γQ∗(st+1,at+1) from the next stage onward. The difference between the two sides of the equality is known as the temporal difference error, δ:

δ = Q(s,a) − (r + γ maxQ(s′,a)) (5) a

Our goal is the minimise this error, so that we can have a good Q function to estimate the rewards given any state-action pair.

Experiments

Open the Colab notebook link to begin: [DQN notebook link]. Read through the notebook and play around with it. More detailed instructions are given in the notebook. Have fun!

1. [1pt] Implementation of ϵ − greedy

Complete the function get action for the agent to select an action based on current state. We want to balance exploitation and exploration through ϵ − greedy, which is explained in the notebook. Include your code snippet in your write-up.

2. [1pt] Implementation of DQN training step

Complete the function train for the model to perform a single step of optimization. This is basically to construct the the temporal difference error δ and perform a standard optimizer update. Notice that there are two networks in the DQN network, policy net and target net, think about how to use these two networks to construct the loss. Include your code snippet in your write-up.

3. [2pt] Train your DQN Agent

After implementing the required functions, now you can train your DQN Agent, and you are suggested to tune the hyperparameters listed in the notebook. Hyperparameters are important to train a good agent. After all of these, now you can validate your model by playing the CartPole Balance game! List the hyperparameters’ value you choose, your epsilon decay rule and summarize your final results from the visualizations in a few sentences in your write-up.

What you need to submit

• Your code files: a4-dcgan.ipynb, a4-gcn.ipynb, a4-dqn.ipynb.

• A PDF document titled a4-writeup.pdf containing code screenshots, any experiment results or visualizations, as well as your answers to the written questions.

Further Resources

1. Generative Adversarial Nets (Goodfellow et al., 2014)

5. An Introduction to GANs in Tensorflow

6. Generative Models Blog Post from OpenAI

References

Prithviraj Sen, Galileo Mark Namata, Mustafa Bilgic, Lise Getoor, Brian Gallagher, and Tina Eliassi-Rad. Collective classification in network data. AI Magazine, 29(3):93–106, 2008. URL http://www.cs.iit.edu/~ml/pdfs/sen-aimag08.pdf.

URL https://openreview.net/forum?id=rJXMpikCZ. accepted as poster.
