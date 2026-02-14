# =========================
# LABEL ENCODER
# =========================

LabelEncoder <- setRefClass(
  "LabelEncoder",
  fields=list(map="list"),

  methods=list(

    fit=function(x){
      vals <- unique(x)
      map <<- setNames(seq_along(vals)-1, vals)
    },

    transform=function(x){
      as.numeric(map[x])
    },

    fit_transform=function(x){
      fit(x)
      transform(x)
    }

  )
)


# =========================
# ONE HOT ENCODER
# =========================

OneHotEncoder <- setRefClass(
  "OneHotEncoder",
  fields=list(categories="list"),

  methods=list(

    fit=function(df){

      categories <<- lapply(df, function(col)
        unique(col)
      )

    },

    transform=function(df){

      out <- NULL

      for(colname in names(df)){

        col <- df[[colname]]
        cats <- categories[[colname]]

        mat <- sapply(cats, function(val)
          as.integer(col == val)
        )

        colnames(mat) <- paste(colname,cats,sep="_")

        out <- cbind(out, mat)
      }

      as.matrix(out)
    },

    fit_transform=function(df){
      fit(df)
      transform(df)
    }

  )
)
