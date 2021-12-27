command="$(<lambda-labs-setup.sh)"
ssh -i ~/mlab_ssh $1 "$command"