mod.ufmg.comsimul.rodada <- function(tab, p = 5, comp.nome = NULL, n = 10000, nr = 2, plot = FALSE) {
  
  # Criar o nome do arquivo dinamicamente
  arquivo_saida <- paste0("prev.mod.ufmg.comsimul.rodada", comp.nome, ".csv")
  
  # Garantir que uma simulação não impacte outra
  times_status <- times_status0
  
  # Ordenar tab pela rodada e ano
  tab1 <- tab |> arrange(ano, rodada)
  
  # Construir o vetor de índice
  tab_rod <- tab1 |>
    mutate(rod_ano = paste(ano, rodada)) |>
    mutate(indice = as.numeric(factor(rod_ano, levels = unique(rod_ano))))
  
  # Previsões para as primeiras rodadas
  for (o in 1:nr) {
    jogos0 <- tab_rod[tab_rod$indice == o, ]
    
    # Faz as previsões
    for (i in 1:nrow(jogos0)) {
      jogo <- jogos0[i, ]
      m <- jogo$mand
      v <- jogo$vist
      
      mand <- times_status[times_status$Times == m, ]
      vist <- times_status[times_status$Times == v, ]
      
      if(is.na(jogo$Camp)){
        
        prev <- data.frame(
          id        = jogo$id,
          ano       = jogo$ano,
          rodada    = jogo$rodada,
          mandante  = mand$Times,
          visitante = vist$Times,
          pvm       = (mand$pvm + vist$pdv)/2,
          pe        = (mand$pem + vist$pev)/2,
          pvv       = (mand$pdm + vist$pvv)/2
        )
        
      }else{
        
        prev <- data.frame(
          id        = jogo$id,
          ano       = jogo$ano,
          rodada    = jogo$rodada,
          mandante  = mand$Times,
          visitante = vist$Times,
          pvm       = (mand$pvm + vist$pdm)/2,
          pe        = (mand$pem + vist$pem)/2,
          pvv       = (mand$pdm + vist$pvm)/2
        )
      }
      
      # Determinar o resultado do modelo
      prev$resultado <- c("vm", "em", "vv")[which.max(c(prev$pvm, prev$pe, prev$pvv))]
      
      if (plot) print(prev)
      
      # Salvar os resultados
      write_csv(
        prev,
        arquivo_saida,
        na = "NA",
        append = (o > 1 || i > 1),
        col_names = !(o > 1 || i > 1) # Apenas na primeira vez inclui cabeçalho
      )
    }
  }
  
  # Processar as rodadas subsequentes
  for (l in (nr + 1):length(unique(tab_rod$indice))) {
    jogos <- tab_rod[tab_rod$indice == l, ]
    jogos_ant <- tab_rod[tab_rod$indice >= (l - nr) & tab_rod$indice < l, ]
    times_status <- times_status0
    
    # Atualizar os parâmetros com base nos jogos anteriores
    for (m in 1:nrow(jogos_ant)) {
      jogo <- jogos_ant[m, ]
      m <- jogo$mand
      v <- jogo$vist
      
      mand <- times_status[times_status$Times == m, ]
      vist <- times_status[times_status$Times == v, ]
      
      PM <- mand[c(2, 3, 4)]
      PV <- vist[c(5, 6, 7)]
      
      # Atualizar parâmetros com base no resultado
      if (jogo$resul == "vm") {
        PM <- ((p * PM) + (vist$R * c(1, 0, 0))) / (p + vist$R)
        PV <- ((p * PV) + ((1 - mand$R) * c(1, 0, 0))) / (p + (1 - mand$R))
        mand$Rc <- mand$Rc + 3
      } else if (jogo$resul == "vv") {
        PM <- ((p * PM) + ((1 - vist$R) * c(0, 0, 1))) / (p + (1 - vist$R))
        PV <- ((p * PV) + (mand$R * c(0, 0, 1))) / (p + mand$R)
        vist$Rc <- vist$Rc + 3
      } else {
        # Empates
        if (jogo$resul == "em") {
          PM <- if (vist$R <= 0.5) {
            (p * PM + (1 - 2 * vist$R) * c(0, 0.5, 0.5) + 2 * vist$R * c(0, 1, 0)) / (p + 1)
          } else {
            (p * PM + (2 * vist$R - 1) * c(0.5, 0.5, 0) + 2 * (1 - vist$R) * c(0, 1, 0)) / (p + 1)
          }
          PV <- if (mand$R <= 0.5) {
            (p * PV + (1 - 2 * mand$R) * c(0, 0.5, 0.5) + 2 * mand$R * c(0, 1, 0)) / (p + 1)
          } else {
            (p * PV + (2 * mand$R - 1) * c(0.5, 0.5, 0) + 2 * (1 - mand$R) * c(0, 1, 0)) / (p + 1)
          }
          mand$Rc <- mand$Rc + 1
          vist$Rc <- vist$Rc + 1
        }
      }
      
      mand$Rd <- mand$Rd + 3
      vist$Rd <- vist$Rd + 3
      mand$R <- mand$Rc / mand$Rd
      vist$R <- vist$Rc / vist$Rd
      
      mand[c(2, 3, 4)] <- PM
      vist[c(5, 6, 7)] <- PV
      
      times_status[times_status$Times == m, ] <- mand
      times_status[times_status$Times == v, ] <- vist
    }
    
    # Realizar previsões para a rodada atual
    for (i in 1:nrow(jogos)) {
      jogo <- jogos[i, ]
      m <- jogo$mand
      v <- jogo$vist
      
      mand <- times_status[times_status$Times == m, ]
      vist <- times_status[times_status$Times == v, ]
      
      prev <- data.frame(
        id = jogo$id,
        ano = jogo$ano,
        rodada = jogo$rodada,
        mandante = mand$Times,
        visitante = vist$Times,
        pvm = (mand$pvm + vist$pdv) / 2,
        pe = (mand$pem + vist$pev) / 2,
        pvv = (mand$pdm + vist$pvv) / 2
      )
      
      numeros <- sort(runif(n, 0, 1))
      n_vm <- sum(numeros <= prev$pvm)
      n_vv <- sum(numeros >= (prev$pvm + prev$pe))
      prev$pvm <- n_vm / n
      prev$pvv <- n_vv / n
      prev$pe <- 1 - (prev$pvm + prev$pvv)
      prev$resultado <- c("vm", "em", "vv")[which.max(c(prev$pvm, prev$pe, prev$pvv))]
      
      if (plot) print(prev)
      
      write_csv(
        prev,
        arquivo_saida,
        na = "NA",
        append = TRUE,
        col_names = FALSE
      )
    }
  }
  
  cat("Data frame", arquivo_saida, "de previsões criado e salvo na pasta.\n")
}
