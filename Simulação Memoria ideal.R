# Descrição: Script para carregar os modelos de previsão de resultados de partidas de futebol e calcular métricas de desempenho.
setwd("~/Documents/Dissertação/scripts")
library(tidyverse)
library(patchwork)
library(ggpattern)
# carregando funções
{
  ### preparação de dados
  
  source("Funções dos modelos/prep_dados.r")
  # parametros: id,dia,ano,rodada,mand,vist,plac_mand,plac_vist
  
  ### ufmg com simulação uniforme
  {
    source("Funções dos modelos/mod.ufmg.comsimul.geral.r")
    # parametros: tab,p,complemento do nome do arquivo, número de simulações, plotar TouF
    source("Funções dos modelos/mod.ufmg.comsimul.ano.r")
    # parametros: tab,p,complemento do nome do arquivo, número de simulações, plotar TouF
    source("Funções dos modelos/mod.ufmg.comsimul.janela.r")
    # parametros: tab,p,complemento do nome do arquivo, número de simulações, plotar TouF
    source("Funções dos modelos/mod.ufmg.comsimul.rodada.r")
    # parametros: tab,p,complemento do nome do arquivo, número de simulações, número de rodadas, plotar TouF
  }
  
  ### ufmg sem simulação
  {
    source("Funções dos modelos/mod.ufmg.semsimul.geral.r")
    # parametros: tab,p,complemento do nome do arquivo, plotar TouF
    source("Funções dos modelos/mod.ufmg.semsimul.ano.r")
    # parametros: tab,p,complemento do nome do arquivo, plotar TouF
    source("Funções dos modelos/mod.ufmg.semsimul.janela.r")
    # parametros: tab,p,complemento do nome do arquivo, plotar TouF
    source("Funções dos modelos/mod.ufmg.semsimul.rodada.r")
    # parametros: tab,p,complemento do nome do arquivo, número de rodadas, plotar TouF
  }
  
  ### ufmg com adição de erro normal
  {
    source("Funções dos modelos/mod.ufmg.simulnormal.geral.r")
    # parametros: tab,p,complemento do nome do arquivo, plotar TouF
    source("Funções dos modelos/mod.ufmg.simulnormal.ano.r")
    # parametros: tab,p,complemento do nome do arquivo, plotar TouF
    source("Funções dos modelos/mod.ufmg.simulnormal.janela.r")
    # parametros: tab,p,complemento do nome do arquivo, plotar TouF
    source("Funções dos modelos/mod.ufmg.simulnormal.rodada.r")
    # parametros: tab,p,complemento do nome do arquivo, número de rodadas, plotar TouF
  }
  
  ### modelos controle
  {
    source("Funções dos modelos/mod.controle.mand.r") 
    # parametros: tab,complemento do nome do arquivo, plotar TouF
    source("Funções dos modelos/mod.controle.emp.r") 
    # parametros: tab,complemento do nome do arquivo, plotar TouF
    source("Funções dos modelos/mod.controle.vist.r") 
    # parametros: tab,complemento do nome do arquivo, plotar TouF
    source("Funções dos modelos/mod.controle.poisson.r")
    # parametros: tab,complemento do nome do arquivo, plotar TouF
    source("Funções dos modelos/mod.controle.uniforme.r")
    # parametros: tab,complemento do nome do arquivo, plotar TouF
    source("Funções dos modelos/mod.controle.implicito.media.r") 
    # parametros: tab,complemento do nome do arquivo, plotar TouF
    source("Funções dos modelos/mod.controle.media.r") 
    # parametros: tab,complemento do nome do arquivo, plotar TouF
  }
  
  ### modelos Arruda
  {
    source("Funções dos modelos/mod.arr.SD0.geral.r") 
    # parametros: tab,complemento do nome do arquivo, plotar TouF
    source("Funções dos modelos/mod.arr.SD0.ano.r") 
    # parametros: tab,complemento do nome do arquivo, plotar TouF
    source("Funções dos modelos/mod.arr.SD0.janela.r") 
    # parametros: tab,complemento do nome do arquivo, plotar TouF
    source("Funções dos modelos/mod.arr.SD0.rodada.r") 
    # parametros: tab,complemento do nome do arquivo, nº de rodadas a considerar, plotar TouF
    
    source("Funções dos modelos/mod.arr.SD1.geral.r") 
    # parametros: tab,complemento do nome do arquivo, plotar TouF
    source("Funções dos modelos/mod.arr.SD1.ano.r") 
    # parametros: tab,complemento do nome do arquivo, plotar TouF
    source("Funções dos modelos/mod.arr.SD1.janela.r") 
    # parametros: tab,complemento do nome do arquivo, plotar TouF
    source("Funções dos modelos/mod.arr.SD1.rodada.r") 
    # parametros: tab,complemento do nome do arquivo, nº de rodadas a considerar, plotar TouF
    
    source("Funções dos modelos/mod.arr.chance1.geral.r") 
    # parametros: tab,complemento do nome do arquivo, plotar TouF
    source("Funções dos modelos/mod.arr.chance1.ano.r") 
    # parametros: tab,complemento do nome do arquivo, plotar TouF
    source("Funções dos modelos/mod.arr.chance1.janela.r") 
    # parametros: tab,complemento do nome do arquivo, plotar TouF
    source("Funções dos modelos/mod.arr.chance1.rodada.r") 
    # parametros: tab,complemento do nome do arquivo, nº de rodadas a considerar, plotar TouF
    
    source("Funções dos modelos/mod.arr.chance2.geral.r") 
    # parametros: tab,complemento do nome do arquivo, plotar TouF
    source("Funções dos modelos/mod.arr.chance2.ano.r") 
    # parametros: tab,complemento do nome do arquivo, plotar TouF
    source("Funções dos modelos/mod.arr.chance2.janela.r") 
    # parametros: tab,complemento do nome do arquivo, plotar TouF
    source("Funções dos modelos/mod.arr.chance2.rodada.r") 
    # parametros: tab,complemento do nome do arquivo, nº de rodadas a considerar, plotar TouF
  }
  
  ### metricas
  {
    source("Funções dos modelos/met.taxa.acerto.r")
    # parametros: previsto, observado, retornar_comparacao = TouF
    source("Funções dos modelos/met.taxa.func.r")
    # parametros: previsto
    source("Funções dos modelos/met.epmp.r")
    # parametros: previsto, observado, pvm, pem, pvv, retornar_comparacao = TouF
    source("Funções dos modelos/met.definetti.r")
    # parametros: pvm, pem, pvv, resul, retornar_componentes = TouF
    source("Funções dos modelos/met.definetti.detalhada.r")
    # parametros: placar_man, placar_vis, prev_golman, prev_golvis, retornar_componentes = TouF
    source("Funções dos modelos/met.matriz.confusao.r")
    # parametros: previsto, observado, retornar_comparacao = TouF
    source("Funções dos modelos/met.vero.r")
    # parametros: placar_man, placar_vis, lambda1, lambda2, retornar_componentes = TouF
  }
  
  br <- read_csv("bancos/br.csv", 
                 col_types = cols(ID = col_number(),
                                  rodata = col_number(), 
                                  data = col_date(format = "%d/%m/%Y"), 
                                  ano = col_number(),
                                  hora = col_time(format = "%H:%M"), 
                                  mandante_Placar = col_number(),
                                  visitante_Placar = col_number()
                 )
  )
  classicos <- read_csv("bancos/classicos.csv")
  attach(br)
  
  prep_dados(ID,data,ano,rodata,mandante,visitante,mandante_Placar,visitante_Placar,classicos)

}

setwd("~/Documents/Dissertação/scripts/simula2")

###  Simulações dos modelos ---------------------------------
tab <- subset(tab, ano %in% c(2021,2022,2023,2024))
tab_sim <- tab

#### rodar as simulações 
{
# mod.arr.SD0.rodada.r
# parametros: tab, complemento do nome do arquivo, nº de rodadas a considerar, plotar T ou F

{
  modelo  <- mod.arr.SD0.rodada
  modname <- "mod.arr.SD0.rodada"
  
  # Arquivo de saída corrigido
  arquivo_saida  <- "Simula_Resultado/simula_mod_arr_SD0.csv"
  arquivo_existe <- file.exists(arquivo_saida)
  
  # Loop de rodadas
  for (rod in 2:114) {
    
    nome <- paste0(".Sim.", rod)
    
    # Executa o modelo
    modelo(tab_sim, nome, rod, FALSE)
    
    # Lê arquivo de previsões
    prev <- readr::read_csv(
      paste0("prev.", modname, nome, ".csv"),
      col_types = readr::cols(
        pvm = readr::col_double(),
        pe  = readr::col_double(),
        pvv = readr::col_double(),
        golman = readr::col_double(),
        golvis = readr::col_double(),
        lambda1 = readr::col_double(),
        lambda2 = readr::col_double(),
        .default = readr::col_guess()
      )
    ) |>
      dplyr::arrange(id) |>
      dplyr::filter(ano == 2024)
    
    tab_comp <- dplyr::filter(tab_sim, ano == 2024)
    
    # Calcula métricas de confusão uma vez só
    cm_full <- met.cm(prev$resultado, tab_comp$resul, FALSE)
    
    # Calcula métricas finais
    resultados_df <- tibble::tibble(
      Mod    = modname,
      Rodada = rod,
      TA     = met.taxa.acerto(prev$resultado, tab_comp$resul, FALSE),
      EPMP   = met.epmp(prev$resultado, tab_comp$resul,
                        prev$pvm, prev$pe, prev$pvv, FALSE),
      MD     = met.definetti(prev$pvm, prev$pe, prev$pvv,
                             tab_comp$resul, FALSE),
      MDD    = met.definetti.detalhada(tab_comp$plac_mand,
                                       tab_comp$plac_vist,
                                       prev$golman, prev$golvis, FALSE),
      TF     = met.taxa.func(prev$resultado),
      
      # Métricas da matriz de confusão
      MC_Pre = cm_full$Precision,
      MC_Rec = cm_full$Recall,
      MC_F1  = cm_full$F1,
      MC_MCC = cm_full$MCC,
      
      # Verossimilhança
      Vero   = met.vero(tab_comp$plac_mand,
                        tab_comp$plac_vist,
                        prev$lambda1, prev$lambda2, FALSE)
    )
    
    # Salva no CSV acumulado
    readr::write_csv(
      resultados_df,
      arquivo_saida,
      na = "NA",
      append = arquivo_existe
    )
    
    # A partir da segunda rodada sempre faz append
    arquivo_existe <- TRUE
  }
}

#mod.arr.SD1.rodada.r
  # parametros: tab,complemento do nome do arquivo, nº de rodadas a considerar, plotar TouF
{
  modelo  <- mod.arr.SD1.rodada
  modname <- "mod.arr.SD1.rodada"
  
  # Arquivo de saída corrigido
  arquivo_saida  <- "Simula_Resultado/simula_mod_arr_SD1.csv"
  arquivo_existe <- file.exists(arquivo_saida)
  
  # Loop de rodadas
  for (rod in 2:114) {
    
    nome <- paste0(".Sim.", rod)
    
    # Executa o modelo
    modelo(tab_sim, nome, rod, FALSE)
    
    # Lê arquivo de previsões
    prev <- readr::read_csv(
      paste0("prev.", modname, nome, ".csv"),
      col_types = readr::cols(
        pvm = readr::col_double(),
        pe  = readr::col_double(),
        pvv = readr::col_double(),
        golman = readr::col_double(),
        golvis = readr::col_double(),
        lambda1 = readr::col_double(),
        lambda2 = readr::col_double(),
        .default = readr::col_guess()
      )
    ) |>
      dplyr::arrange(id) |>
      dplyr::filter(ano == 2024)
    
    tab_comp <- dplyr::filter(tab_sim, ano == 2024)
    
    # Calcula métricas de confusão uma vez só
    cm_full <- met.cm(prev$resultado, tab_comp$resul, FALSE)
    
    # Calcula métricas finais
    resultados_df <- tibble::tibble(
      Mod    = modname,
      Rodada = rod,
      TA     = met.taxa.acerto(prev$resultado, tab_comp$resul, FALSE),
      EPMP   = met.epmp(prev$resultado, tab_comp$resul,
                        prev$pvm, prev$pe, prev$pvv, FALSE),
      MD     = met.definetti(prev$pvm, prev$pe, prev$pvv,
                             tab_comp$resul, FALSE),
      MDD    = met.definetti.detalhada(tab_comp$plac_mand,
                                       tab_comp$plac_vist,
                                       prev$golman, prev$golvis, FALSE),
      TF     = met.taxa.func(prev$resultado),
      
      # Métricas da matriz de confusão
      MC_Pre = cm_full$Precision,
      MC_Rec = cm_full$Recall,
      MC_F1  = cm_full$F1,
      MC_MCC = cm_full$MCC,
      
      # Verossimilhança
      Vero   = met.vero(tab_comp$plac_mand,
                        tab_comp$plac_vist,
                        prev$lambda1, prev$lambda2, FALSE)
    )
    
    # Salva no CSV acumulado
    readr::write_csv(
      resultados_df,
      arquivo_saida,
      na = "NA",
      append = arquivo_existe
    )
    
    # A partir da segunda rodada sempre faz append
    arquivo_existe <- TRUE
  }
}

#mod.arr.chance1.rodada.r
  # parametros: tab,complemento do nome do arquivo, nº de rodadas a considerar, plotar TouF
{
  modelo  <- mod.arr.chance1.rodada
  modname <- "mod.arr.chance1.rodada"
  
  # Arquivo de saída corrigido
  arquivo_saida  <- "Simula_Resultado/simula_mod_arr_chance1.csv"
  arquivo_existe <- file.exists(arquivo_saida)
  
  # Loop de rodadas
  for (rod in 2:114) {
    
    nome <- paste0(".Sim.", rod)
    
    # Executa o modelo
    modelo(tab_sim, nome, rod, FALSE)
    
    # Lê arquivo de previsões
    prev <- readr::read_csv(
      paste0("prev.", modname, nome, ".csv"),
      col_types = readr::cols(
        pvm = readr::col_double(),
        pe  = readr::col_double(),
        pvv = readr::col_double(),
        golman = readr::col_double(),
        golvis = readr::col_double(),
        lambda1 = readr::col_double(),
        lambda2 = readr::col_double(),
        .default = readr::col_guess()
      )
    ) |>
      dplyr::arrange(id) |>
      dplyr::filter(ano == 2024)
    
    tab_comp <- dplyr::filter(tab_sim, ano == 2024)
    
    # Calcula métricas de confusão uma vez só
    cm_full <- met.cm(prev$resultado, tab_comp$resul, FALSE)
    
    # Calcula métricas finais
    resultados_df <- tibble::tibble(
      Mod    = modname,
      Rodada = rod,
      TA     = met.taxa.acerto(prev$resultado, tab_comp$resul, FALSE),
      EPMP   = met.epmp(prev$resultado, tab_comp$resul,
                        prev$pvm, prev$pe, prev$pvv, FALSE),
      MD     = met.definetti(prev$pvm, prev$pe, prev$pvv,
                             tab_comp$resul, FALSE),
      MDD    = met.definetti.detalhada(tab_comp$plac_mand,
                                       tab_comp$plac_vist,
                                       prev$golman, prev$golvis, FALSE),
      TF     = met.taxa.func(prev$resultado),
      
      # Métricas da matriz de confusão
      MC_Pre = cm_full$Precision,
      MC_Rec = cm_full$Recall,
      MC_F1  = cm_full$F1,
      MC_MCC = cm_full$MCC,
      
      # Verossimilhança
      Vero   = met.vero(tab_comp$plac_mand,
                        tab_comp$plac_vist,
                        prev$lambda1, prev$lambda2, FALSE)
    )
    
    # Salva no CSV acumulado
    readr::write_csv(
      resultados_df,
      arquivo_saida,
      na = "NA",
      append = arquivo_existe
    )
    
    # A partir da segunda rodada sempre faz append
    arquivo_existe <- TRUE
  }
}

#mod.arr.chance2.rodada.r
  # parametros: tab,complemento do nome do arquivo, nº de rodadas a considerar, plotar TouF
{
  modelo  <- mod.arr.chance2.rodada
  modname <- "mod.arr.chance2.rodada"
  
  # Arquivo de saída corrigido
  arquivo_saida  <- "Simula_Resultado/simula_mod_arr_chance2.csv"
  arquivo_existe <- file.exists(arquivo_saida)
  
  # Loop de rodadas
  for (rod in 2:114) {
    
    nome <- paste0(".Sim.", rod)
    
    # Executa o modelo
    modelo(tab_sim, nome, rod, FALSE)
    
    # Lê arquivo de previsões
    prev <- readr::read_csv(
      paste0("prev.", modname, nome, ".csv"),
      col_types = readr::cols(
        pvm = readr::col_double(),
        pe  = readr::col_double(),
        pvv = readr::col_double(),
        golman = readr::col_double(),
        golvis = readr::col_double(),
        lambda1 = readr::col_double(),
        lambda2 = readr::col_double(),
        .default = readr::col_guess()
      )
    ) |>
      dplyr::arrange(id) |>
      dplyr::filter(ano == 2024)
    
    tab_comp <- dplyr::filter(tab_sim, ano == 2024)
    
    # Calcula métricas de confusão uma vez só
    cm_full <- met.cm(prev$resultado, tab_comp$resul, FALSE)
    
    # Calcula métricas finais
    resultados_df <- tibble::tibble(
      Mod    = modname,
      Rodada = rod,
      TA     = met.taxa.acerto(prev$resultado, tab_comp$resul, FALSE),
      EPMP   = met.epmp(prev$resultado, tab_comp$resul,
                        prev$pvm, prev$pe, prev$pvv, FALSE),
      MD     = met.definetti(prev$pvm, prev$pe, prev$pvv,
                             tab_comp$resul, FALSE),
      MDD    = met.definetti.detalhada(tab_comp$plac_mand,
                                       tab_comp$plac_vist,
                                       prev$golman, prev$golvis, FALSE),
      TF     = met.taxa.func(prev$resultado),
      
      # Métricas da matriz de confusão
      MC_Pre = cm_full$Precision,
      MC_Rec = cm_full$Recall,
      MC_F1  = cm_full$F1,
      MC_MCC = cm_full$MCC,
      
      # Verossimilhança
      Vero   = met.vero(tab_comp$plac_mand,
                        tab_comp$plac_vist,
                        prev$lambda1, prev$lambda2, FALSE)
    )
    
    # Salva no CSV acumulado
    readr::write_csv(
      resultados_df,
      arquivo_saida,
      na = "NA",
      append = arquivo_existe
    )
    
    # A partir da segunda rodada sempre faz append
    arquivo_existe <- TRUE
  }
}

#mod.ufmg.comsimul.rodada.r
  # parametros: tab,p,complemento do nome do arquivo, número de simulações, número de rodadas, plotar TouF
{
  modelo <- mod.ufmg.comsimul.rodada
  modname <- "mod.ufmg.comsimul.rodada"
  # Define arquivo de saída
  arquivo_saida  <- "Simula_Resultado/simula_mod_ufmg_comsimul.csv"
  arquivo_existe <- file.exists(arquivo_saida)
  
  # Loop de rodadas
  for (rod in 2:114) {
    
    nome <- paste0(".Sim.", rod)
    
    # Executa o modelo
    modelo(tab_sim, 5, nome, 1000, rod, FALSE)
    
    # Lê arquivo de previsões
    prev <- readr::read_csv(
      paste0("prev.", modname, nome, ".csv"),
      col_types = readr::cols(
        pvm = readr::col_double(),
        pe  = readr::col_double(),
        pvv = readr::col_double(),
        golman = readr::col_double(),
        golvis = readr::col_double(),
        lambda1 = readr::col_double(),
        lambda2 = readr::col_double(),
        .default = readr::col_guess()
      )
    ) |>
      dplyr::arrange(id) |>
      dplyr::filter(ano == 2024)
    
    tab_comp <- dplyr::filter(tab_sim, ano == 2024)
    
    # Calcula métricas de confusão uma vez só
    cm_full <- met.cm(prev$resultado, tab_comp$resul, FALSE)
    
    # Calcula métricas finais
    resultados_df <- tibble::tibble(
      Mod    = modname,
      Rodada = rod,
      TA     = met.taxa.acerto(prev$resultado, tab_comp$resul, FALSE),
      EPMP   = met.epmp(prev$resultado, tab_comp$resul,  prev$pvm, prev$pe, prev$pvv, FALSE),
      #MD     = met.definetti(prev$pvm, prev$pe, prev$pvv, tab_comp$resul, FALSE),
      #MDD    = met.definetti.detalhada(tab_comp$plac_mand, tab_comp$plac_vist, prev$golman, prev$golvis, FALSE),
      #Vero   = met.vero(tab_comp$plac_mand,tab_comp$plac_vist,prev$lambda1, prev$lambda2, FALSE),
      TF     = met.taxa.func(prev$resultado),
      MC_Pre = cm_full$Precision,
      MC_Rec = cm_full$Recall,
      MC_F1  = cm_full$F1,
      MC_MCC = cm_full$MCC
      
    )
    
    # Salva no CSV acumulado
    readr::write_csv(
      resultados_df,
      arquivo_saida,
      na = "NA",
      append = arquivo_existe
    )
    
    # A partir da segunda rodada sempre faz append
    arquivo_existe <- TRUE
  }
}

#mod.ufmg.semsimul.rodada.r
  # parametros: tab,p,complemento do nome do arquivo, número de rodadas, plotar TouF
{modelo <- mod.ufmg.semsimul.rodada
  modname <- "mod.ufmg.semsimul.rodada"
  # Define arquivo de saída
  arquivo_saida  <- "Simula_Resultado/simula_mod_ufmg_semsimul.csv"
  arquivo_existe <- file.exists(arquivo_saida)
    # Loop de rodadas
    for (rod in 2:114) {
      
      nome <- paste0(".Sim.", rod)
      
      # Executa o modelo
      modelo(tab_sim, 5, nome, rod, FALSE)
      
      # Lê arquivo de previsões
      prev <- readr::read_csv(
        paste0("prev.", modname, nome, ".csv"),
        col_types = readr::cols(
          pvm = readr::col_double(),
          pe  = readr::col_double(),
          pvv = readr::col_double(),
          golman = readr::col_double(),
          golvis = readr::col_double(),
          lambda1 = readr::col_double(),
          lambda2 = readr::col_double(),
          .default = readr::col_guess()
        )
      ) |>
        dplyr::arrange(id) |>
        dplyr::filter(ano == 2024)
      
      tab_comp <- dplyr::filter(tab_sim, ano == 2024)
      
      # Calcula métricas de confusão uma vez só
      cm_full <- met.cm(prev$resultado, tab_comp$resul, FALSE)
      
      # Calcula métricas finais
      resultados_df <- tibble::tibble(
        Mod    = modname,
        Rodada = rod,
        TA     = met.taxa.acerto(prev$resultado, tab_comp$resul, FALSE),
        EPMP   = met.epmp(prev$resultado, tab_comp$resul,  prev$pvm, prev$pe, prev$pvv, FALSE),
        #MD     = met.definetti(prev$pvm, prev$pe, prev$pvv, tab_comp$resul, FALSE),
        #MDD    = met.definetti.detalhada(tab_comp$plac_mand, tab_comp$plac_vist, prev$golman, prev$golvis, FALSE),
        #Vero   = met.vero(tab_comp$plac_mand,tab_comp$plac_vist,prev$lambda1, prev$lambda2, FALSE),
        TF     = met.taxa.func(prev$resultado),
        MC_Pre = cm_full$Precision,
        MC_Rec = cm_full$Recall,
        MC_F1  = cm_full$F1,
        MC_MCC = cm_full$MCC
        
      )
      
      # Salva no CSV acumulado
      readr::write_csv(
        resultados_df,
        arquivo_saida,
        na = "NA",
        append = arquivo_existe
      )
      
      # A partir da segunda rodada sempre faz append
      arquivo_existe <- TRUE
    }
  }

#mod.ufmg.simulnormal.rodada.r
  # parametros: tab,p,complemento do nome do arquivo, número de rodadas, plotar TouF
{modelo <- mod.ufmg.simulnormal.rodada
  modname <- "mod.ufmg.simulnormal.rodada"
  # Define arquivo de saída
  arquivo_saida  <- "Simula_Resultado/simula_mod_ufmg_simulnormal.csv"
  arquivo_existe <- file.exists(arquivo_saida)
  # Loop de rodadas
  for (rod in 2:114) {
    
    nome <- paste0(".Sim.", rod)
    
    # Executa o modelo
    modelo(tab_sim, 5, nome, rod, FALSE)
    
    # Lê arquivo de previsões
    prev <- readr::read_csv(
      paste0("prev.", modname, nome, ".csv"),
      col_types = readr::cols(
        pvm = readr::col_double(),
        pe  = readr::col_double(),
        pvv = readr::col_double(),
        golman = readr::col_double(),
        golvis = readr::col_double(),
        lambda1 = readr::col_double(),
        lambda2 = readr::col_double(),
        .default = readr::col_guess()
      )
    ) |>
      dplyr::arrange(id) |>
      dplyr::filter(ano == 2024)
    
    tab_comp <- dplyr::filter(tab_sim, ano == 2024)
    
    # Calcula métricas de confusão uma vez só
    cm_full <- met.cm(prev$resultado, tab_comp$resul, FALSE)
    
    # Calcula métricas finais
    resultados_df <- tibble::tibble(
      Mod    = modname,
      Rodada = rod,
      TA     = met.taxa.acerto(prev$resultado, tab_comp$resul, FALSE),
      EPMP   = met.epmp(prev$resultado, tab_comp$resul,  prev$pvm, prev$pe, prev$pvv, FALSE),
      #MD     = met.definetti(prev$pvm, prev$pe, prev$pvv, tab_comp$resul, FALSE),
      #MDD    = met.definetti.detalhada(tab_comp$plac_mand, tab_comp$plac_vist, prev$golman, prev$golvis, FALSE),
      #Vero   = met.vero(tab_comp$plac_mand,tab_comp$plac_vist,prev$lambda1, prev$lambda2, FALSE),
      TF     = met.taxa.func(prev$resultado),
      MC_Pre = cm_full$Precision,
      MC_Rec = cm_full$Recall,
      MC_F1  = cm_full$F1,
      MC_MCC = cm_full$MCC
      
    )
    
    # Salva no CSV acumulado
    readr::write_csv(
      resultados_df,
      arquivo_saida,
      na = "NA",
      append = arquivo_existe
    )
    
    # A partir da segunda rodada sempre faz append
    arquivo_existe <- TRUE
  }
}
}

#### Analise dos dados obtidos ---------------------------------
# Leitura dos resultados
{setwd("~/Documents/Dissertação/scripts/simula2/Simula_Resultado")

simula_mod_arr_chance1 <- read_csv("simula_mod_arr_chance1.csv",
                                   col_types = cols(.default = col_double())) |>
  mutate(Mod = "Chance1") |> 
  select(Mod,Rodada,TA,MD,MC_Pre,MC_Rec,MC_F1,MC_MCC)

simula_mod_arr_chance2 <- read_csv("simula_mod_arr_chance2.csv",
  col_types = cols(.default = col_double())) |> 
  mutate(Mod = "Chance2") |> 
  select(Mod,Rodada,TA,MD,MC_Pre,MC_Rec,MC_F1,MC_MCC)

simula_mod_arr_SD0 <- read_csv("simula_mod_arr_SD0.csv",
  col_types = cols(.default = col_double())) |> 
  mutate(Mod = "SD0")|> 
  select(Mod,Rodada,TA,MD,MC_Pre,MC_Rec,MC_F1,MC_MCC)

simula_mod_arr_SD1 <- read_csv("simula_mod_arr_SD1.csv",
  col_types = cols(.default = col_double())) |> 
  mutate(Mod = "SD1") |> 
  select(Mod,Rodada,TA,MD,MC_Pre,MC_Rec,MC_F1,MC_MCC)

simula_mod_ufmg_comsimul <- read_csv("simula_mod_ufmg_comsimul.csv",
  col_types = cols(.default = col_double())) |> 
  mutate(Mod = "Ufmg") |> 
  select(Mod,Rodada,TA,MC_Pre,MC_Rec,MC_F1,MC_MCC)

simula_mod_ufmg_semsimul <- read_csv("simula_mod_ufmg_semsimul.csv",
  col_types = cols(.default = col_double())) |> 
  mutate(Mod = "UfmgSS") |> 
  select(Mod,Rodada,TA,MC_Pre,MC_Rec,MC_F1,MC_MCC)

simula_mod_ufmg_simulnormal <- read_csv("simula_mod_ufmg_simulnormal.csv",
  col_types = cols(.default = col_double())) |> 
  mutate(Mod = "UfmgSn") |> 
  select(Mod,Rodada,TA,MC_Pre,MC_Rec,MC_F1,MC_MCC)
}

#### Taxa de acerto
{graf_agulha <- function(df, titulo = "Modelo", gCor) {
  ggplot(df, aes(x = Rodada, y = TA)) +
    geom_point(color = gCor, size = 2) +
    labs(
      title = titulo,
      subtitle = "Taxa de Acerto por Rodada",
      x = "Número de Rodadas",
      y = "TA"
    ) +
    scale_x_continuous(breaks = seq(min(df$Rodada), max(df$Rodada), by = 20)) +
    scale_y_continuous(breaks = seq(0.2, 0.5, by = 0.1),
                       limits = c(0.1, 0.6)) +
    theme_minimal(base_size = 10)
}
  
  g_SD0      <- graf_agulha(simula_mod_arr_SD0,      "SD0", "aquamarine")
  g_SD1      <- graf_agulha(simula_mod_arr_SD1,      "SD1", "turquoise4")
  g_ch1      <- graf_agulha(simula_mod_arr_chance1,  "Chance 1", "slateblue")
  g_ch2      <- graf_agulha(simula_mod_arr_chance2,  "Chance 2", "dodgerblue3")
  g_UFMG     <- graf_agulha(simula_mod_ufmg_comsimul,     "UFMG", "brown")
  g_UFMG_SS  <- graf_agulha(simula_mod_ufmg_semsimul,  "UFMG SS", "goldenrod2")
  g_UFMG_Sn  <- graf_agulha(simula_mod_ufmg_simulnormal,  "UFMG Sn", "chocolate3")
  
  g.ta.rod<- (g_SD0  | g_SD1 )    /
             (g_ch1  | g_ch2)     /
             (g_UFMG | g_UFMG_SS )/
                 g_UFMG_Sn 
  
    ggsave("graf_TA_rod.png", plot = g.ta.rod,
                width = 15, height = 18, units = "cm", dpi = 300)
}

#### MD
{graf_agulha <- function(df, titulo = "Modelo", gCor) {
  ggplot(df, aes(x = Rodada, y = MD)) +
    geom_point(color = gCor, size = 2) +
    labs(
      title = titulo,
      subtitle = "Medida de DeFinetti por Rodada",
      x = "Número de Rodadas",
      y = "MD"
    ) +
    scale_x_continuous(breaks = seq(min(df$Rodada), max(df$Rodada), by = 20)) +
    scale_y_continuous(breaks = seq(0.5, 0.9, by = 0.1),
                       limits = c(0.5, 0.92)) +
    theme_minimal(base_size = 10)
}
  
  g_SD0      <- graf_agulha(simula_mod_arr_SD0,      "SD0", "aquamarine")
  g_SD1      <- graf_agulha(simula_mod_arr_SD1,      "SD1", "turquoise4")
  g_ch1      <- graf_agulha(simula_mod_arr_chance1,  "Chance 1", "slateblue")
  g_ch2      <- graf_agulha(simula_mod_arr_chance2,  "Chance 2", "dodgerblue3")
#  g_UFMG     <- graf_agulha(simula_mod_ufmg_comsimul,     "UFMG", "brown")
#  g_UFMG_SS  <- graf_agulha(simula_mod_ufmg_semsimul,  "UFMG SS", "goldenrod2")
#  g_UFMG_Sn  <- graf_agulha(simula_mod_ufmg_simulnormal,  "UFMG Sn", "chocolate3")
  
  g.md.rod <- (g_SD0  | g_SD1)/
             (g_ch1  | g_ch2)
  
  ggsave("graf_md_rod.png", plot = g.md.rod,
         width = 15, height = 12, units = "cm", dpi = 300)
}

#### precisão
{graf_agulha <- function(df, titulo = "Modelo", gCor) {
  ggplot(df, aes(x = Rodada, y = MC_Pre)) +
    geom_point(color = gCor, size = 2) +
    labs(
      title = titulo,
      subtitle = "Medida de Precisão por Rodada",
      x = "Número de Rodadas",
      y = "Precisão"
    ) +
    scale_x_continuous(breaks = seq(min(df$Rodada), max(df$Rodada), by = 20)) +
    scale_y_continuous(breaks = seq(0.2, 0.6, by = 0.1),
                       limits = c(0.2, 0.65)) +
    theme_minimal(base_size = 10)
}
  
  g_SD0      <- graf_agulha(simula_mod_arr_SD0,      "SD0", "aquamarine")
  g_SD1      <- graf_agulha(simula_mod_arr_SD1,      "SD1", "turquoise4")
  g_ch1      <- graf_agulha(simula_mod_arr_chance1,  "Chance 1", "slateblue")
  g_ch2      <- graf_agulha(simula_mod_arr_chance2,  "Chance 2", "dodgerblue3")
  g_UFMG     <- graf_agulha(simula_mod_ufmg_comsimul,     "UFMG", "brown")
  g_UFMG_SS  <- graf_agulha(simula_mod_ufmg_semsimul,  "UFMG SS", "goldenrod2")
  g_UFMG_Sn  <- graf_agulha(simula_mod_ufmg_simulnormal,  "UFMG Sn", "chocolate3")
  
  g.pre.rod<- (g_SD0  | g_SD1 )    /
    (g_ch1  | g_ch2)     /
    (g_UFMG | g_UFMG_SS )/
    g_UFMG_Sn 
  
  ggsave("graf_mc.pre_rod.png", plot = g.pre.rod,
         width = 15, height = 18, units = "cm", dpi = 300)
}

#### recall
{graf_agulha <- function(df, titulo = "Modelo", gCor) {
  ggplot(df, aes(x = Rodada, y = MC_Rec)) +
    geom_point(color = gCor, size = 2) +
    labs(
      title = titulo,
      subtitle = "Medida de Recall por Rodada",
      x = "Número de Rodadas",
      y = "Recall"
    ) +
    scale_x_continuous(breaks = seq(min(df$Rodada), max(df$Rodada), by = 20)) +
    scale_y_continuous(breaks = seq(0.2, 0.4, by = 0.1),
                       limits = c(0.2, 0.45)) +
    theme_minimal(base_size = 10)
}
  
  g_SD0      <- graf_agulha(simula_mod_arr_SD0,      "SD0", "aquamarine")
  g_SD1      <- graf_agulha(simula_mod_arr_SD1,      "SD1", "turquoise4")
  g_ch1      <- graf_agulha(simula_mod_arr_chance1,  "Chance 1", "slateblue")
  g_ch2      <- graf_agulha(simula_mod_arr_chance2,  "Chance 2", "dodgerblue3")
  g_UFMG     <- graf_agulha(simula_mod_ufmg_comsimul,     "UFMG", "brown")
  g_UFMG_SS  <- graf_agulha(simula_mod_ufmg_semsimul,  "UFMG SS", "goldenrod2")
  g_UFMG_Sn  <- graf_agulha(simula_mod_ufmg_simulnormal,  "UFMG Sn", "chocolate3")
  
  g.rec.rod<- (g_SD0  | g_SD1 )    /
    (g_ch1  | g_ch2)     /
    (g_UFMG | g_UFMG_SS )/
    g_UFMG_Sn 
  
  ggsave("graf_mc.rec_rod.png", plot = g.rec.rod,
         width = 15, height = 18, units = "cm", dpi = 300)
}

#### F1
{graf_agulha <- function(df, titulo = "Modelo", gCor) {
  ggplot(df, aes(x = Rodada, y = MC_F1)) +
    geom_point(color = gCor, size = 2) +
    labs(
      title = titulo,
      subtitle = "F1-Score por Rodada",
      x = "Número de Rodadas",
      y = "F1"
    ) +
    scale_x_continuous(breaks = seq(min(df$Rodada), max(df$Rodada), by = 20)) +
    scale_y_continuous(breaks = seq(0.2, 0.4, by = 0.1),
                       limits = c(0.2, 0.45)) +
    theme_minimal(base_size = 10)
}
  
  g_SD0      <- graf_agulha(simula_mod_arr_SD0,      "SD0", "aquamarine")
  g_SD1      <- graf_agulha(simula_mod_arr_SD1,      "SD1", "turquoise4")
  g_ch1      <- graf_agulha(simula_mod_arr_chance1,  "Chance 1", "slateblue")
  g_ch2      <- graf_agulha(simula_mod_arr_chance2,  "Chance 2", "dodgerblue3")
  g_UFMG     <- graf_agulha(simula_mod_ufmg_comsimul,     "UFMG", "brown")
  g_UFMG_SS  <- graf_agulha(simula_mod_ufmg_semsimul,  "UFMG SS", "goldenrod2")
  g_UFMG_Sn  <- graf_agulha(simula_mod_ufmg_simulnormal,  "UFMG Sn", "chocolate3")
  
  g.f1.rod<- (g_SD0  | g_SD1 )    /
    (g_ch1  | g_ch2)     /
    (g_UFMG | g_UFMG_SS )/
    g_UFMG_Sn 
  
  ggsave("graf_mc.f1_rod.png", plot = g.f1.rod,
         width = 15, height = 18, units = "cm", dpi = 300)
}

#### MCC
{graf_agulha <- function(df, titulo = "Modelo", gCor) {
  ggplot(df, aes(x = Rodada, y = MC_MCC)) +
    geom_point(color = gCor, size = 2) +
    labs(
      title = titulo,
      subtitle = "Coeficiente de Matthews  por Rodada",
      x = "Número de Rodadas",
      y = "MCC"
    ) +
    scale_x_continuous(breaks = seq(min(df$Rodada), max(df$Rodada), by = 20)) +
    scale_y_continuous(breaks = seq(-0.2, 0.3, by = 0.1),
                       limits = c(-0.2, 0.4)) +
    theme_minimal(base_size = 10)
}
  
  g_SD0      <- graf_agulha(simula_mod_arr_SD0,      "SD0", "aquamarine")
  g_SD1      <- graf_agulha(simula_mod_arr_SD1,      "SD1", "turquoise4")
  g_ch1      <- graf_agulha(simula_mod_arr_chance1,  "Chance 1", "slateblue")
  g_ch2      <- graf_agulha(simula_mod_arr_chance2,  "Chance 2", "dodgerblue3")
  g_UFMG     <- graf_agulha(simula_mod_ufmg_comsimul,     "UFMG", "brown")
  g_UFMG_SS  <- graf_agulha(simula_mod_ufmg_semsimul,  "UFMG SS", "goldenrod2")
  g_UFMG_Sn  <- graf_agulha(simula_mod_ufmg_simulnormal,  "UFMG Sn", "chocolate3")
  
  g.mcc.rod<- (g_SD0  | g_SD1 )    /
    (g_ch1  | g_ch2)     /
    (g_UFMG | g_UFMG_SS )/
    g_UFMG_Sn 
  
  ggsave("graf_mcc_rod.png", plot = g.mcc.rod,
         width = 15, height = 18, units = "cm", dpi = 300)
}
