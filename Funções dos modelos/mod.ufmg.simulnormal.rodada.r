
mod.ufmg.simulnormal.rodada <- function(tab, p = 5, comp.nome = NULL, nr = 2, plot = F){
  
  # Criar o nome do arquivo dinamicamente
  arquivo_saida <- paste0("prev.mod.ufmg.simulnormal.rodada", comp.nome, ".csv")
  
  # Verificar se o arquivo já existe
  arquivo_existe <- file.exists(arquivo_saida)
  
  # garanti que uma modelo não impacte no outro
  times_status <- times_status0
  
  # Ordenar tab pela rodada e ano
  tab1 <- tab |> arrange(ano, rodada)
  
  # Construir o vetor de índice
  tab_rod <- tab1 |>
    mutate(rod_ano = paste(ano, rodada)) |>
    mutate(indice = as.numeric(factor(rod_ano, levels = unique(rod_ano))))
  
  #fazer previsões para os primeiros jogos
  for (o in 1:nr) {
    
    jogos0 <- tab_rod[tab_rod$indice == o,]
    
    # faz as previsões
    for (i in 1:nrow(jogos0)) {
      
      jogo <- jogos0[i,]
      
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
      
      # Salvar o arquivo com o nome dinâmico
      write_csv(
        prev,
        arquivo_saida,
        na = "NA",
        append = arquivo_existe,  # Append dados se arquivo existe
        col_names = !arquivo_existe  # Cabeçalhos apenas na primeira gravação
      )
      
      arquivo_existe <- TRUE # Atualizar o estado do arquivo
    }
  }
  
  ## fragmenta em rodadas
  for (l in (nr+1):length(unique(tab_rod$indice))) {
    
    
    jogos <- tab_rod[tab_rod$indice==l,]
    jogos_ant <- tab_rod[(tab_rod$indice>=(l-nr) & tab_rod$indice<l),]
    times_status <- times_status0
    
    # atualiza os parametros
    for (m in 1:nrow(jogos_ant)) {
      # atualização dos parametros 
      
      
      jogo <- jogos_ant[m,]
      
      m <- jogo$mand
      v <- jogo$vist
      
      mand <- times_status[times_status$Times == m, ]
      vist <- times_status[times_status$Times == v, ]
      
      PM <- mand[c(2,3,4)]
      PV <- vist[c(5,6,7)]
      
      # caso vitoria mandante ##########
      if(jogo$resul == "vm"){
        PM <- ((p * PM) + (vist$R * c(1,0,0)))/ (p+vist$R)
        
        PV <- ((p * PV) + ((1-mand$R) * c(1,0,0)))/ (p+(1-mand$R))
        
        mand$Rc <-mand$Rc+3
        
      }else{
        # caso vitoria visitante ###########
        if(jogo$resul == "vv"){
          PM <- ((p * PM) + ((1-vist$R) * c(0,0,1)))/ (p+(1-vist$R))
          
          PV <- ((p * PV) + (mand$R * c(0,0,1)))/ (p+mand$R)
          
          vist$Rc <-vist$Rc+3
          
        }else{
          # caso empate ################
          if(jogo$resul == "em" & vist$R<=0.5){
            ## caso Rv <= 0.5
            PM <-  (p * PM + (1 - 2 * vist$R) * c(0, 0.5, 0.5) + 2 * vist$R * c(0, 1, 0)) / (p + 1)
            
          }else{
            ## caso Rv > 0.5 
            PM <- (p * PM + (2 * vist$R - 1) * c(0.5, 0.5, 0) + 2 * (1 - vist$R) * c(0, 1, 0)) / (p + 1)
            
          }
          # caso empate #################
          if(jogo$resul == "em" & mand$R<=0.5){
            ## caso Rm <= 0.5
            PV <-  (p * PV + (1 - 2 * mand$R) * c(0, 0.5, 0.5) + 2 * mand$R * c(0, 1, 0)) / (p + 1)
            
          }else{
            ## caso Rm > 0.5 
            PV <- (p * PV + (2 * mand$R - 1) * c(0.5, 0.5, 0) + 2 * (1 - mand$R) * c(0, 1, 0)) / (p + 1)
            
          }
          mand$Rc <-mand$Rc + 1
          vist$Rc <-vist$Rc + 1
        }
      }
      
      mand$Rd <- mand$Rd+3
      vist$Rd <- vist$Rd+3
      mand$R  <- mand$Rc/mand$Rd
      vist$R  <- vist$Rc/vist$Rd
      
      mand[c(2,3,4)] <- PM
      vist[c(5,6,7)] <- PV
      
      times_status[times_status$Times == m, ] <- mand
      times_status[times_status$Times == v, ] <- vist
      assign("times_status", times_status, envir = .GlobalEnv)
    }
    
    # faz as previsões
    for (i in 1:nrow(jogos)) {
      
      jogo <- jogos[i,]
      
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
      
      # Adiciona erro normal
      
      prev$pvm <- prev$pvm + rnorm(1,0,sqrt(0.0001))
      prev$pvv <- prev$pvv + rnorm(1,0,sqrt(0.0001))
      prev$pe  <- 1-(prev$pvm+prev$pvv)
      
      # determinando resultado do modelo
      
      prev$resultado <- ifelse(max(prev$pvm,prev$pe,prev$pvv) == prev$pvm, "vm", 
                               ifelse(max(prev$pvm,prev$pe,prev$pvv) == prev$pvv, "vv", "em"))
      if(plot==T){print(prev)}
      
      # Salvar o arquivo com o nome dinâmico
      write_csv(
        prev,
        arquivo_saida,
        na = "NA",
        append = TRUE
      )
      
    }
  }
  cat("Data frame", arquivo_saida," de previsões criado e salvo na pasta.\n")
  
}


