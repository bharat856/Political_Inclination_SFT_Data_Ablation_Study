.libPaths("/fs/scratch/users/bharatpi/causal_political_content/envs/Rlibs")

library(manifestoR)

mp_setapikey(key = "e56edd3182588c6b8874de96ca1c124d")

sample_path <- "/fs/scratch/users/bharatpi/causal_political_content/data/raw/manifesto/manifesto_download_sample_30_corpus_available.csv"
out_dir <- "/fs/scratch/users/bharatpi/causal_political_content/data/raw/manifesto/corpus"

dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

sample <- read.csv(sample_path)

wanted <- data.frame(
  party = sample$party,
  date = sample$date
)

corpus <- mp_corpus(wanted)

saveRDS(corpus, file.path(out_dir, "sample_30_manifesto_corpus.rds"))

print(corpus)