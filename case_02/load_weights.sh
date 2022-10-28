gdown "1-8c-DpxJ6iOAnHjOqyWvnbWvUUlxYxK6" -O ./weights/
unzip ./weights/weights_sbert.zip -d ./weights/
rm -rf ./weights/weights_sbert.zip

gdown "1AB7lawi9G4gv1WAG6ZVwnS-HgU4JkWiK" -O ./weights/embeddings.pkl
