met.vero <- function(placar_mand_obs, placar_visit_obs, lambda1, lambda2, retornar_comparacao = FALSE) {
  # Verifica se os vetores têm o mesmo comprimento
  if (!all(length(placar_mand_obs) == length(placar_visit_obs),
           length(lambda1) == length(placar_mand_obs),
           length(lambda2) == length(placar_mand_obs))) {
    stop("Todos os vetores devem ter o mesmo comprimento.")
  }
  
  # Remove valores NA
  tabela <- na.omit(data.frame(
    mand = placar_mand_obs,
    visit = placar_visit_obs,
    lambda1 = lambda1,
    lambda2 = lambda2
  ))
  
  # Cálculo da verossimilhança
  verossimilhanca <- dpois(tabela$mand, tabela$lambda1) *
    dpois(tabela$visit, tabela$lambda2)
  
  # Verossimilhança média
  veross_media <- mean(verossimilhanca)
  
  # Retornar a tabela com todos os valores
  if (retornar_comparacao) {
    comparacao <- cbind(tabela, verossimilhanca = verossimilhanca)
    return(list(
      veross_media = veross_media,
      comparacao = comparacao
    ))
  }
  
  # Caso contrário, retorna apenas a verossimilhança média
  return(veross_media)
}

