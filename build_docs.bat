@echo off
echo Generating documentation...
Rscript -e "setwd('E:/VectorForgeML'); roxygen2::roxygenise(); cat('Docs built\n')"
pause
