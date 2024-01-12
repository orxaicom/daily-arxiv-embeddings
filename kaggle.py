!pip install -U adapters
!git clone "https://github.com/orxaicom/daily-arxiv-embeddings.git"
!python ./daily-arxiv-embeddings/scrape.py
!python ./daily-arxiv-embeddings/embed.py
