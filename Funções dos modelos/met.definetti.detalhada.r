met.definetti.detalhada <- function(placar_man, placar_vis, prev_golman, prev_golvis, retornar_componentes = FALSE) {
  # Verifica se os vetores têm o mesmo comprimento
  if (!all(length(placar_man) == length(placar_vis), 
           length(placar_vis) == length(prev_golman), 
           length(prev_golman) == length(prev_golvis))) {
    stop("Todos os vetores devem ter o mesmo comprimento.")
  }
  
  # Remove valores NA
  tabela <- na.omit(data.frame(
    placar_man  = placar_man,
    placar_vis  = placar_vis,
    prev_golman = prev_golman,
    prev_golvis = prev_golvis
  ))
  
  # Cálculo vetorizado de DF (Distância Euclidiana entre placares reais e previstos)
  tabela$DF <- sqrt((tabela$placar_man - tabela$prev_golman)^2 + 
                      (tabela$placar_vis - tabela$prev_golvis)^2)
  
  # Calcula a média dos valores de DF
  media_DF <- mean(tabela$DF)
  
  # Retorna a média e a tabela de componentes, se solicitado
  if (retornar_componentes) {
    return(list(
      media_DF = media_DF,
      componentes = tabela
    ))
  }
  
  # Retorna apenas a média dos valores de DF
  return(media_DF)
}
