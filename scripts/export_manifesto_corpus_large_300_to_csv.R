.libPaths("/fs/scratch/users/bharatpi/causal_political_content/envs/Rlibs")

library(manifestoR)

corpus_path <- "/fs/scratch/users/bharatpi/causal_political_content/data/raw/manifesto/corpus/large_300_manifesto_corpus.rds"
out_path <- "/fs/scratch/users/bharatpi/causal_political_content/data/raw/manifesto/corpus/large_300_manifesto_corpus.csv"

corpus <- readRDS(corpus_path)

doc_ids <- names(corpus)

texts <- sapply(corpus, function(doc) {
  paste(as.character(doc), collapse = " ")
})

df <- data.frame(
  doc_id = doc_ids,
  text = texts,
  stringsAsFactors = FALSE
)

write.csv(df, out_path, row.names = FALSE)

print(paste("Saved:", out_path))
print(dim(df))
