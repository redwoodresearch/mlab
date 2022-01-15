echo "$1"
command="$(<lambda-labs-setup.sh)"
ssh -o StrictHostKeyChecking=no -i ~/mlab_ssh $1 "$command"