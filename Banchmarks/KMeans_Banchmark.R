library(VectorForgeML)
ls("package:VectorForgeML")

cat("Loading dataset...\n")

df <- read.csv("inst/dataset/winequality.csv",sep=";")

X <- df
X$quality <- NULL


# -----------------------------
# preprocessing
# -----------------------------
scaler <- StandardScaler$new()
X_scaled <- scaler$fit_transform(X)


# -----------------------------
# training
# -----------------------------
cat("Training KMeans...\n")

start <- Sys.time()

model <- KMeans$new(6)
labels <- model$fit_predict(X_scaled)

end <- Sys.time()

cat("Train time:", as.numeric(end-start),"sec\n")


# -----------------------------
# cluster stats
# -----------------------------
cat("\nCluster counts:\n")
print(table(labels))


# -----------------------------
# prediction timing
# -----------------------------
cat("\nTesting prediction speed...\n")

start <- Sys.time()
pred <- model$predict(X_scaled)
end <- Sys.time()

cat("Prediction time:", as.numeric(end-start),"sec\n")


# =========================================================
# VISUALIZATION SECTION
# =========================================================

cat("\nGenerating plots...\n")

# ---------- PCA projection for 2D visualization ----------
pca <- PCA$new(2)
X2 <- pca$fit_transform(X_scaled)

cols <- rainbow(length(unique(labels)))


# ---------- Scatter plot clusters ----------
png("cluster_scatter.png",800,600)

plot(X2[,1],X2[,2],
     col=cols[labels],
     pch=19,
     xlab="PC1",
     ylab="PC2",
     main="KMeans Clusters (PCA Projection)")

legend("topright",
       legend=unique(labels),
       col=cols,
       pch=19)

dev.off()



# ---------- Cluster size barplot ----------
png("cluster_sizes.png",700,500)

barplot(table(labels),
        col=cols,
        main="Cluster Size Distribution",
        xlab="Cluster",
        ylab="Count")

dev.off()



# =========================================================
# ELBOW METHOD GRAPH
# =========================================================

cat("Calculating inertia curve...\n")

inertia <- c()

for(k in 1:10){

  km <- KMeans$new(k)
  lab <- km$fit_predict(X_scaled)

  s <- 0
  for(i in 1:nrow(X_scaled)){
    c <- lab[i]
    center <- km$ptr$centroids[((c-1)*ncol(X_scaled)+1):(c*ncol(X_scaled))]
    s <- s + sum((X_scaled[i,]-center)^2)
  }

  inertia <- c(inertia,s)
}

png("elbow_plot.png",700,500)

plot(1:10,inertia,
     type="b",
     pch=19,
     xlab="K",
     ylab="Inertia",
     main="Elbow Method")

dev.off()



# =========================================================
# SIMPLE CLUSTER SEPARATION SCORE
# =========================================================

cat("Computing separation score...\n")

centroids <- matrix(model$ptr$centroids,
                    nrow=6, byrow=TRUE)

distances <- c()

for(i in 1:nrow(X_scaled)){
  c <- labels[i]
  distances <- c(distances,
                 sqrt(sum((X_scaled[i,]-centroids[c,])^2)))
}

png("cluster_spread.png",700,500)

boxplot(distances ~ labels,
        col=cols,
        main="Cluster Compactness",
        xlab="Cluster",
        ylab="Distance to centroid")

dev.off()


cat("\nAll plots saved successfully.\n")
