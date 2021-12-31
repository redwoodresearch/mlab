TODO: figure out how we want to teach conceptual stuff/how we want to lecture this.
I found spinning up to be pretty good: https://spinningup.openai.com/en/latest/spinningup/rl_intro.html 
Probably a bunch of other possible resources.

# DQN
Need: explanation of greedy selection for Q function to achieve policy which converges (probably link something?)

Now guide to gym

Then I think dirt simple DQN first trains on every state? (This seems right). (Nope actually go straight
to normal buffer implementation)

Don't worry about performance, don't even worry too much about vectorization of operations!

Make sure people see tradeoff between different eps values/do tuning on them.
Subset selection considerations (actually sample without replacement!).
Make sure rendering occurs to see progress (just last episode).

Get people to try running multiple time (with final param values? initial param values?)

Make sure people understand how eps can cause failure on cartpole so you need decay.

Fixed LR.

Actually get people to spend a while messing with params on cartpole. I think this
is pretty worthwhile.

Then afterword, mess with params on beamrider on fast runs and then just let it
run for a while (beamrider is very easy to see some progress in).


