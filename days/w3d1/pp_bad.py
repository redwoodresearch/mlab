    # rank 0  : prv_activations == batch
    # rank 1+ : prv_activations == None
    #
    # rank *  : activations == None

    # Forward
    # [1, 1] --- f1     ---> [2, 4]
    #
    # [2, 4] --- 3a + b ---> 10
    #
    # Backward
    #
    # [6, 4] <--- f1     --- [3, 1]
    #
    # [3, 1] <--- 3a + b --- 1

    # rank -1:
    #   activations == None
    #   prv_activations == Var(act[-2])
    #
    # rank i in 1, ..., -2:
    #   activations == Var(act[i])
    #   prv_activations == Var(act[i - 1])
    #
    # rank 0:
    #   activations == Var(act[0])
    #   prv_activations == batch
