met.taxa.func <- function(previsto) {
  
  # Define os valores que você quer excluir
  valores_excluir <- c("vm", "vv", "em")
  
  # Cria vetor lógico de comparação
  comparacao <- (previsto %in% valores_excluir)
  
  # Calcula a taxa de valores diferentes
  taxa_func <- sum(comparacao, na.rm = TRUE) / length(previsto)

    return(taxa_func)
}
