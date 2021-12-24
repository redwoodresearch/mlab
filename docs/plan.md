## Week 1: PyTorch
In the following, BYO means "build you own"
- Array programming 1.
	- Key concept: tensors are a pointer to some reference-counted memory, and shape and stride, and offset.
	- indexing. masking. broadcasting. arange. einsum. slicing. reshaping. binary operations. reductions. linspace.
		- Maybe https://github.com/Kyubyong/pytorch_exercises? Maybe too easy?
	- scatter/gather/index_select?
	- Vectorized algorithm problems? Vectorized two-sum?
	- Implement a linear regression.
	- logistic regression for MNIST
	- "in re machine learning boot camp, I think that randomly-initialized MLPs that take in (i,j) pixel coordinates and return (R, G, B) values for generating pictures are a pretty fun early task for learning pytorch (doesn't require any optimization or data). It's how I got my profile picture."
	- raytracer?
	- Projects:
		- Do those 100 exercises. https://github.com/rougier/numpy-100/blob/master/100_Numpy_exercises.md
	- [einops](https://github.com/arogozhnikov/einops)
		- http://einops.rocks/pytorch-examples.html
- Array programming 2: more complicated NN
	- implement conv1d and conv2d using pytorch primitives.
		- https://jott.live/markdown/as_strided
	- Implement all the things you need for resnet18: Conv2d, BatchNorm2d, ReLU, MaxPool2d, AdaptiveAvgPool2d, Linear, init, GroupNorm, Sequential
		- [BatchNorm2d](https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/batchnorm.py) not too bad
		- [ReLU](https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/activation.py#L63) is trivial
		- [maxpool2d](https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/pooling.py#L93)
			- surely this is just like a conv but with a max instead of an einsum
			- someone else has https://gist.github.com/nishnik/f9041eff871d269eada6decb9454b5d2
		- AdaptiveAvgPool2d: this just means that you compute the stride and kernel size based on the input size. Equations from SO [here](https://stackoverflow.com/a/58694174).
		- [GroupNorm](https://pytorch.org/docs/stable/generated/torch.nn.GroupNorm.html) is just some kind of norm thing
	- Implement self-attention, other components from transformers
		- Fuck, the HF implementation of GPT-2 is [so absurdly long](https://github.com/huggingface/transformers/blob/master/src/transformers/models/gpt2/modeling_gpt2.py). But the [TF implementation](https://github.com/openai/gpt-2/blob/master/src/model.py) seems way more reasonable. I wonder if there's a shorter implementation of GPT-2.
- nn.Module
	- Implement stateful versions of the things that you've already implemented as functions on previous days
	- nn.ParameterList, nn.ModuleList, nn.BufferList
-  Training neural nets.
	-  Key question: are we supposed to spend time on explaining backprop?
	-  `.backward()`, `.grad`
	- BYO Optimizer: SGD, momentum, Adam
		- https://github.com/geohot/tinygrad/blob/master/tinygrad/optim.py
		- https://ruder.io/optimizing-gradient-descent/
		- takes much less than a day.
- GPUs and remote machines.
	- ssh, tmux, jupyter w/ ngrok
	- try putting your previous stuff on a GPU, it should work faster
	- BYO data parallelism?
[Tinygrad](https://github.com/geohot/tinygrad) is maybe useful
- Can we do an activity where someone randomly changes something about your code such it no longer does the same thing, and you have to figure out what went wrong by using a network visualization tool?
	
https://cs231n.github.io/
Would they find it helpful to use mypy and black?
https://www.tensorflow.org/tutorials/generative/deepdream
## Week 2: Implementing transformers
Resources:
- https://huggingface.co/course/chapter1
- https://nostalgebraist.tumblr.com/post/185326092369/1-classic-fully-connected-neural-networks-these
- https://nostalgebraist.tumblr.com/post/188730918209/huh-the-alphastar-paper-is-finally-up-linked-in
- GPT-2 paper.
- Implement GPT-2. Check that it matches. Check that it performs similarly if you drop it in as a replacement in a HF training loop.
	- https://github.com/openai/gpt-2/blob/master/src/model.py
- Implement BERT. Check that it does the same thing as HF implementation, including gradients.
	- https://github.com/huggingface/transformers/blob/5b317f7ea4fcd012d537f0a1e3c73aef82f9b70a/examples/research_projects/movement-pruning/emmental/modeling_bert_masked.py
	- Hmm. This is kind of a lot of code. The main part that is hard is BertSelfAttention.
	- Shorter implementation: https://github.com/codertimo/BERT-pytorch
- DeBERTa.
	- maybe we just get them to implement the key smart parts, rather than trying to get all the details of hooking things together correct.
- BYO Tokenizer. The key subword tokenizing function is [here](https://github.com/huggingface/transformers/blob/5b317f7ea4fcd012d537f0a1e3c73aef82f9b70a/src/transformers/models/bert/tokenization_bert.py#L509)
	- I claim that this is worth people's time.
	- It's really easy to test this.
- Sampler.
	- https://towardsdatascience.com/how-to-sample-from-language-models-682bceb97277
	- Batched sampler
	- Sampler for pair of contexts.
	- Sampler for "-- and that was bad"
		- efficiently
- Gwern shit
	- https://www.alignmentforum.org/posts/k7oxdbNaGATZbtEg3/redwood-research-s-current-project?commentId=CMvfybQCk6sxtLFMN
## Week 3: Training transformers
The key skill from this week is training transformers.
- Implementing changes to them.
- Running experiments
- Knowing whether an experiment worked.
- Experience with random facts about how to sometimes make transformers train better.
	- Maybe there should be an exercise where you read a pair of papers and comment on the main differences between how they trained.
Days:
- [Gin](https://github.com/google/gin-config). Logging stuff. Queues of jobs. Log graphs of gradients etc. ask smart friends for things here.
	- What's the best way to save data? Mb Redis
- Maybe we should have a single "make a badass train loop" day:
		- Model saving/resuming
		- Metrics
		- Data loading in good ways?
- Fine tune on language modelling. Hyperparam search on LR.
	- Analysis--paired t-test? Other things here?
	- What examples did it perform worst on?
- Perf as fn of size. Scaling laws.
- Dropout. Batch size. Regularization.
- Read papers and figure out what they did?
- Day 1
	- Comet.ml or whatever
	- Hyperparameter search
		- We need to be able to try out different things. Eg LRs, optimizers, architectures. And so what we are going to do:
			- Make your stuff configurable with gin.
			- Make a main method that takes a gin conf file and some overrides on the command line and then runs the job with that.
			- Make search, with
				- create_search_jobs
				- work
				- run_one_job_from_queue
	- FYI this is way more of a PITA if you're using many machines.
	- Run a hyperparam search on some thing
		- look at a paper to get advice on how to set the hyperparams
- Day 2
	- Analyzing results
		- exploratory analysis
		- [Evan's A/B testing tools](https://www.evanmiller.org/ab-testing/)
## Week 4: RL
readings:
- https://nostalgebraist.tumblr.com/post/629020418641199104/this-post-is-an-adaptedextended-version-of-an-lw
- Train a policy to say positive-sentiment, low KL things
	- Or maybe just train a value fn?
	- batching
- Train a two-player game on that task, with just a value fn.
	- [[question]] for DMZ: why don't we use DQN etc for LM tasks?
- FUDGE?
- active learning?
- some of Rainbow DQN?
	- on some game that is really fast to win at?
Things this is missing that I feel sad about:
- SQL
	- i wonder if there's some more obvious data storage system that suffices for all our needs.
- Graphs day
	- maybe this is on a weekend?
- More analysis
- writing good tests
- reading papers
- streamlet?
- alignment content
- flask apps
I'm worried that compared to the "RR bootcamp" this is missing practical stuff like web stuff, graph stuff, etc.
Required knowledge on the way in:
- pytorch
- tmux
- some editor
Other weeks:
- Images
- Atari RL
- Interpretability
- Advex
- Web
- GPU perf/programming
- Unix bullshit
Alternative later week: Generative models. GAN, VAE, diffusion models, autoregressive models.
We could rip off exercises from this course: http://www.cs.toronto.edu/~rgrosse/courses/csc321_2018/
IMO the only way to make this actually work is to have me first build out some code, and then build all the dependencies for it.
Projects to build out:
-
einops.rocks
Writing better code with pytorch and einops
Learning by example: rewriting and fixing popular code fragments (Not automatically expanded because 3 MB is too large. You can expand it anyway or open it in a new window.)