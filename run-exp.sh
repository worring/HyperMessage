#!/bin/bash
export PATH=/home/worring/hypermsg-anaconda/bin:$PATH
python hypermsg.py --data cocitation --dataset cora --split 10 --epochs 250 --addself --cuda --out cora-cocitation
python hypermsg.py --data cocitation --dataset citeseer --split 10 --epochs 250 --addself --cuda --out citeseer-cocitation
python hypermsg.py --data cocitation --dataset pubmed --split 10 --epochs 250 --addself --cuda --out pubmed-cocitation
python hypermsg.py  --data coauthorship --dataset dblp --epochs 250 --split 10  --cuda --generatesplit --shuffle --out dblp-coauthor
python hypermsg.py  --data coauthorship --dataset cora --epochs 250 --split 10  --cuda --generatesplit --shuffle --out cora-coauthor 
python hypermsg.py --inputdir "./data-multimedia" --dataset MIRFLICKR --epochs 250 --split 10 --cuda --generatesplit --shuffle --out multimedia-exp
