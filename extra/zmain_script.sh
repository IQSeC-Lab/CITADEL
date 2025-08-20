TS=$(date "+%m.%d-%H.%M.%S")
nohup python -u zmain.py > zmain_centroid_embedding_out_$TS.log 2>&1 &
