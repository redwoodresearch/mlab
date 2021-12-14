TODO: figure out how we want to teach conceptual stuff/how we want to lecture this.
I found spinning up to be pretty good: https://spinningup.openai.com/en/latest/spinningup/rl_intro.html 
Probably a bunch of other possible resources.

# DQN
Need: explanation of greedy selection for Q function to achieve policy which converges (probably link something?)

Now guide to gym

Then I think dirt simple DQN first trains on every state? (This seems right).

One big batch, don't worry at all about efficiency.
Don't worry about performance even for vectorization of operations!
(I'm a bit worried about how this plays with rapid feedback, maybe just don't train to
covergences?)

Make sure people see tradeoff between different eps values/do tuning on them.
Subset selection considerations (actually sample without replacement!).
Make sure rendering occurs to see progress (just last episode).

Get people to try running multiple time (with final param values? initial param values?)

Make sure people understand how eps can cause failure on cartpole.

Also try decaying eps value and decaying learning rate?

Actually get people to spend a while messing with params on cartpole. I think this
is pretty worthwhile.

Then afterword, mess with params on beamrider on fast runs and then just let it
run for a while (beamrider is very easy to see some progress in).


