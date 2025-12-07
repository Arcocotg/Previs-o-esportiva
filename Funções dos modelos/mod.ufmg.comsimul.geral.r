mod.ufmg.comsimul.geral <- function(tab, p = 5, comp.nome = NULL, n = 10000, plot = F) {
  
  # Criar o nome do arquivo dinamicamente
  arquivo_saida <- paste0("prev.mod.ufmg.comsimul.geral", comp.nome, ".csv")
  
  # Verificar se o arquivo já existe
  arquivo_existe <- file.exists(arquivo_saida)
  
  for (i in 1:nrow(tab)) {
    
    # Previsão
    jogo <- tab[i, ]
    
    m <- jogo$mand
    v <- jogo$vist
    
    mand <- times_status[times_status$Times == m, ]
    vist <- times_status[times_status$Times == v, ]
    
    if (is.na(jogo$Camp)) {
      prev <- data.frame(
        id        = jogo$id,
        ano       = jogo$ano,
        rodada    = jogo$rodada,
        mandante  = mand$Times,
        visitante = vist$Times,
        pvm       = (mand$pvm + vist$pdv) / 2,
        pe        = (mand$pem + vist$pev) / 2,
        pvv       = (mand$pdm + vist$pvv) / 2
      )
    } else {
      prev <- data.frame(
        id        = jogo$id,
        ano       = jogo$ano,
        rodada    = jogo$rodada,
        mandante  = mand$Times,
        visitante = vist$Times,
        pvm       = (mand$pvm + vist$pdm) / 2,
        pe        = (mand$pem + vist$pem) / 2,
        pvv       = (mand$pdm + vist$pvm) / 2
      )
    }
    
    # Simulação da proporção em cada intervalo
    numeros <- sort(runif(n, min = 0, max = 1))
    n_vm <- subset(numeros, numeros <= prev$pvm)
    n_vv <- subset(numeros, numeros >= (prev$pvm + prev$pe))
    prev$pvm <- length(n_vm) / n
    prev$pvv <- length(n_vv) / n
    prev$pe <- (n - length(c(n_vm, n_vv))) / n
    
    # Determinando resultado do modelo
    prev$resultado <- ifelse(
      max(prev$pvm, prev$pe, prev$pvv) == prev$pvm, "vm",
      ifelse(max(prev$pvm, prev$pe, prev$pvv) == prev$pvv, "vv", "em")
    )
    if (plot == T) {
      print(prev)
    }
    
    # Salvar o arquivo com cabeçalhos apenas na primeira escrita
    write_csv(
      prev,
      arquivo_saida,
      na = "NA",
      append = arquivo_existe, # Append dados se arquivo existe
      col_names = !arquivo_existe # Cabeçalhos apenas na primeira escrita
    )
    
    arquivo_existe <- TRUE  # Atualizar o estado do arquivo
  }
  
  cat("Data frame", arquivo_saida, "de previsões criado e salvo na pasta.\n")
}
