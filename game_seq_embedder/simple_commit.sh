# bash simple_commit.sh 1 origin master

msg=${1:-1}
dst=${2:-'origin'}
branch=${3:-'master'}

python app/cp_py_file.py
bash cp_transformers.sh
git add -A
git commit -am $msg
git push $dst $branch

