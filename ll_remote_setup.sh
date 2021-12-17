command="$(<lambda-labs-setup.sh)"
ssh -i ~/.ssh/mlab $1 "$command"