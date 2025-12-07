mod.controle.poisson <- function(tab, comp.nome = NULL, plot = F) {
  
  # Criar o nome do arquivo dinamicamente
  arquivo_saida <- paste0("prev.mod.controle.poisson", comp.nome, ".csv")
  
  # Verificar se o arquivo já existe
  arquivo_existe <- file.exists(arquivo_saida)
  
  # Previsão
  for (i in 1:nrow(tab)) {
    jogo2 <- tab[i, ]
    
    prev <- data.frame(
      id        = jogo2$id,
      ano       = jogo2$ano,
      rodada    = jogo2$rodada,
      mandante  = jogo2$mand,
      visitante = jogo2$vist,
      golman = rpois(1, 1),
      golvis = rpois(1, 1)
    )
    
    prev$resultado <- ifelse(prev$golman > prev$golvis, "vm", 
                             ifelse(prev$golman == prev$golvis, "em", "vv"))
    
    if (plot == T) {
      print(prev)
    }
    
    # Salvar o arquivo com cabeçalho apenas se for a primeira vez
    write_csv(
      prev,
      arquivo_saida,
      na = "NA",
      append = arquivo_existe, # Controla a inclusão do cabeçalho
      col_names = !arquivo_existe # Inclui cabeçalho apenas se o arquivo ainda não existe
    )
    
    # Atualizar o status do arquivo para as próximas iterações
    arquivo_existe <- TRUE
  }
  
  cat("Data frame", arquivo_saida, "de previsões criado e salvo na pasta.\n")
}
