sudo apt install default-jre 
sudo apt install openjdk-8-jre-headless
pip3 install numpy scipy sklearn pandas networkx gensim nltk

mkdir -p results/confusion/

python3 create_scripts.py $2 $3
chmod -R 777 scripts
nohup ./scripts/fila_bow/fila_bow.sh &