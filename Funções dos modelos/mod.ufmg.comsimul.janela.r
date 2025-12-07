mod.ufmg.comsimul.janela <- function(tab, p = 5, comp.nome = NULL, n = 10000, plot = F){
  
  # Criar o nome do arquivo dinamicamente
  arquivo_saida <- paste0("prev.mod.ufmg.comsimul.janela", comp.nome, ".csv")
  
  # Ordenar tab pelo dia
  tab <- tab |> arrange(dia)
  
  # Semestre atual
  if(format(tab$dia[1] , "%m") < "07" ){
    semetre_atual <- 1
  } else {
    semetre_atual <- 2
  } 
  
  for (i in 1:nrow(tab)) {
    
    # Previsão
    jogo <- tab[i,]
    
    # Semestre do jogo
    if(format(jogo$dia , "%m") < "07" ){
      semestre <- 1
    } else {
      semestre <- 2
    }
    
    # Verifica se mudou de semestre
    if(semestre != semetre_atual){
      times_status <- times_status0
      semetre_atual <- semestre
    }
    
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
    } else {
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
    
    # Simulação da proporção em cada intervalo
    numeros <- sort(runif(n, min = 0, max = 1))
    
    n_vm <- subset(numeros, numeros <= prev$pvm)
    n_vv <- subset(numeros, numeros >= (prev$pvm + prev$pe))
    
    prev$pvm <- length(n_vm) / n
    prev$pvv <- length(n_vv) / n
    prev$pe  <- (n - length(c(n_vm, n_vv))) / n
    
    # Determinando resultado do modelo
    prev$resultado <- ifelse(max(prev$pvm, prev$pe, prev$pvv) == prev$pvm, "vm", 
                             ifelse(max(prev$pvm, prev$pe, prev$pvv) == prev$pvv, "vv", "em"))
    if(plot == T){ print(prev) }
    
    # Salvar o arquivo com o nome dinâmico
    if (i == 1 && !file.exists(arquivo_saida)) {
      write_csv(
        prev,
        arquivo_saida,
        na = "NA",
        append = FALSE,
        col_names = TRUE
      )
    } else {
      write_csv(
        prev,
        arquivo_saida,
        na = "NA",
        append = TRUE,
        col_names = FALSE
      )
    }
    
    # Atualização dos parâmetros 
    PM <- mand[c(2, 3, 4)]
    PV <- vist[c(5, 6, 7)]
    
    if(jogo$resul == "vm"){ 
      PM <- ((p * PM) + (vist$R * c(1, 0, 0))) / (p + vist$R)
      PV <- ((p * PV) + ((1 - mand$R) * c(1, 0, 0))) / (p + (1 - mand$R))
      mand$Rc <- mand$Rc + 3
    } else if(jogo$resul == "vv"){
      PM <- ((p * PM) + ((1 - vist$R) * c(0, 0, 1))) / (p + (1 - vist$R))
      PV <- ((p * PV) + (mand$R * c(0, 0, 1))) / (p + mand$R)
      vist$Rc <- vist$Rc + 3
    } else {
      if(jogo$resul == "em" & vist$R <= 0.5){
        PM <- (p * PM + (1 - 2 * vist$R) * c(0, 0.5, 0.5) + 2 * vist$R * c(0, 1, 0)) / (p + 1)
      } else {
        PM <- (p * PM + (2 * vist$R - 1) * c(0.5, 0.5, 0) + 2 * (1 - vist$R) * c(0, 1, 0)) / (p + 1)
      }
      if(jogo$resul == "em" & mand$R <= 0.5){
        PV <- (p * PV + (1 - 2 * mand$R) * c(0, 0.5, 0.5) + 2 * mand$R * c(0, 1, 0)) / (p + 1)
      } else {
        PV <- (p * PV + (2 * mand$R - 1) * c(0.5, 0.5, 0) + 2 * (1 - mand$R) * c(0, 1, 0)) / (p + 1)
      }
      mand$Rc <- mand$Rc + 1
      vist$Rc <- vist$Rc + 1
    }
    
    mand$Rd <- mand$Rd + 3
    vist$Rd <- vist$Rd + 3
    mand$R  <- mand$Rc / mand$Rd
    vist$R  <- vist$Rc / vist$Rd
    
    mand[c(2, 3, 4)] <- PM
    vist[c(5, 6, 7)] <- PV
    
    times_status[times_status$Times == m, ] <- mand
    times_status[times_status$Times == v, ] <- vist
    assign("times_status", times_status, envir = .GlobalEnv)
  }
  
  cat("Data frame", arquivo_saida, "de previsões criado e salvo na pasta.\n")
}
