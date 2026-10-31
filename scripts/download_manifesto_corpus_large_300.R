.libPaths("/fs/scratch/users/bharatpi/causal_political_content/envs/Rlibs")

library(manifestoR)

mp_setapikey(key = "e56edd3182588c6b8874de96ca1c124d")

input_path <- "/fs/scratch/users/bharatpi/causal_political_content/data/raw/manifesto/manifesto_download_large_300.csv"
out_dir <- "/fs/scratch/users/bharatpi/causal_political_content/data/raw/manifesto/corpus"

dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

sample <- read.csv(input_path)

wanted <- data.frame(
  party = sample$party,
  date = sample$date
)

print(paste("Requested documents:", nrow(wanted)))

corpus <- mp_corpus(wanted)

saveRDS(corpus, file.path(out_dir, "large_300_manifesto_corpus.rds"))

print(corpus)
print(paste("Saved:", file.path(out_dir, "large_300_manifesto_corpus.rds")))
