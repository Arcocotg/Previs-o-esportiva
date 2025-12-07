# Descrição: Script para carregar os modelos de previsão de resultados de partidas de futebol e calcular métricas de desempenho.
setwd("~/Documents/Dissertação/scripts/Previsões")
setwd("~/Documents/Dissertação/scripts/teste2")
library(tidyverse, patchwork, ggpattern)
library(kableExtra)

# ler tabelas dos modelos 
{ prev.mod.controle.mand <- read_csv("prev.mod.controle.mand.csv",
                                     col_types = cols(pvm = col_number(), 
                                                      pe = col_number(), 
                                                      pvv = col_number()),
                                     col_names = T)
  prev.mod.controle.mand <- arrange(prev.mod.controle.mand, id)
  
  prev.mod.controle.vist <- read_csv("prev.mod.controle.vist.csv",
                                     col_types = cols(pvm = col_number(), 
                                                      pe = col_number(), 
                                                      pvv = col_number()),
                                     col_names = T)
  prev.mod.controle.vist <- arrange(prev.mod.controle.vist, id)
  
  prev.mod.controle.emp <- read_csv("prev.mod.controle.emp.csv",
                                     col_types = cols(pvm = col_number(), 
                                                      pe = col_number(), 
                                                      pvv = col_number()),
                                     col_names = T)
  prev.mod.controle.emp <- arrange(prev.mod.controle.emp, id)

  prev.mod.controle.poisson <- read_csv("prev.mod.controle.poisson.csv",
                                     col_types = cols(golman = col_number(), 
                                                      golvis = col_number()),
                                     col_names = T)
  prev.mod.controle.poisson <- arrange(prev.mod.controle.poisson, id)

  prev.mod.controle.uniforme <- read_csv("prev.mod.controle.uniforme.csv",
                                         col_types = cols(pvm = col_number(), 
                                                          pe = col_number(), 
                                                          pvv = col_number()),
                                         col_names = T)
  prev.mod.controle.uniforme <- arrange(prev.mod.controle.uniforme, id)

  prev.mod.controle.implicito.media <- read_csv("prev.mod.controle.implicito.media.csv",
                                                col_types = cols(pvm = col_number(), 
                                                                 pe = col_number(), 
                                                                 pvv = col_number(),
                                                                 golman = col_number(), 
                                                                 golvis = col_number()),
                                                col_names = T)
  prev.mod.controle.implicito.media <- arrange(prev.mod.controle.implicito.media, id)
  
  prev.mod.controle.media <- read_csv("prev.mod.controle.media.csv",
                                      col_types = cols(golman = col_number(), 
                                                       golvis = col_number()),
                                      col_names = T)
  prev.mod.controle.media <- arrange(prev.mod.controle.media, id)
  
  #####
  prev.mod.ufmg.comsimul.geral <- read_csv("prev.mod.ufmg.comsimul.geral.csv",
                                           col_types = cols(pvm = col_number(), 
                                                            pe = col_number(), 
                                                            pvv = col_number()),
                                           col_names = T)
  prev.mod.ufmg.comsimul.geral <- arrange(prev.mod.ufmg.comsimul.geral, id)
  
  prev.mod.ufmg.comsimul.ano <- read_csv("prev.mod.ufmg.comsimul.ano.csv",
                                         col_types = cols(pvm = col_number(), 
                                                          pe = col_number(), 
                                                          pvv = col_number()),
                                         col_names = T)
  prev.mod.ufmg.comsimul.ano <- arrange(prev.mod.ufmg.comsimul.ano, id)
  
  prev.mod.ufmg.comsimul.janela <- read_csv("prev.mod.ufmg.comsimul.janela.csv",
                                            col_types = cols(pvm = col_number(), 
                                                             pe = col_number(), 
                                                             pvv = col_number()),
                                            col_names = T)
  prev.mod.ufmg.comsimul.janela <- arrange(prev.mod.ufmg.comsimul.janela, id)
  
  prev.mod.ufmg.comsimul.rodada <- read_csv("prev.mod.ufmg.comsimul.rodada.csv",
                                            col_types = cols(pvm = col_number(), 
                                                             pe = col_number(), 
                                                             pvv = col_number()),
                                            col_names = T)
  prev.mod.ufmg.comsimul.rodada <- arrange(prev.mod.ufmg.comsimul.rodada, id)
  
  #####
  prev.mod.ufmg.semsimul.geral <- read_csv("prev.mod.ufmg.semsimul.geral.csv",
                                           col_types = cols(pvm = col_number(), 
                                                            pe = col_number(), 
                                                            pvv = col_number()),
                                           col_names = T)
  prev.mod.ufmg.semsimul.geral <- arrange(prev.mod.ufmg.semsimul.geral, id)
  
  prev.mod.ufmg.semsimul.ano <- read_csv("prev.mod.ufmg.semsimul.ano.csv",
                                         col_types = cols(pvm = col_number(), 
                                                          pe = col_number(), 
                                                          pvv = col_number()),
                                         col_names = T)
  prev.mod.ufmg.semsimul.ano <- arrange(prev.mod.ufmg.semsimul.ano, id)
 
  prev.mod.ufmg.semsimul.janela <- read_csv("prev.mod.ufmg.semsimul.janela.csv",
                                            col_types = cols(pvm = col_number(), 
                                                             pe = col_number(), 
                                                             pvv = col_number()),
                                            col_names = T)
  prev.mod.ufmg.semsimul.janela <- arrange(prev.mod.ufmg.semsimul.janela, id)
  
  prev.mod.ufmg.semsimul.rodada <- read_csv("prev.mod.ufmg.semsimul.rodada.csv",
                                            col_types = cols(pvm = col_number(), 
                                                             pe = col_number(), 
                                                             pvv = col_number()),
                                            col_names = T)
  prev.mod.ufmg.semsimul.rodada <- arrange(prev.mod.ufmg.semsimul.rodada, id)
  
  #####
  prev.mod.ufmg.simulnormal.geral <- read_csv("prev.mod.ufmg.simulnormal.geral.csv",
                                              col_types = cols(pvm = col_number(), 
                                                               pe = col_number(), 
                                                               pvv = col_number()),
                                              col_names = T)
  prev.mod.ufmg.simulnormal.geral <- arrange(prev.mod.ufmg.simulnormal.geral, id)
  
  prev.mod.ufmg.simulnormal.ano <- read_csv("prev.mod.ufmg.simulnormal.ano.csv",
                                            col_types = cols(pvm = col_number(), 
                                                             pe = col_number(), 
                                                             pvv = col_number()),
                                            col_names = T)
  prev.mod.ufmg.simulnormal.ano <- arrange(prev.mod.ufmg.simulnormal.ano, id)

  prev.mod.ufmg.simulnormal.janela <- read_csv("prev.mod.ufmg.simulnormal.janela.csv",
                                               col_types = cols(pvm = col_number(), 
                                                                pe = col_number(), 
                                                                pvv = col_number()),
                                               col_names = T)
  prev.mod.ufmg.simulnormal.janela <- arrange(prev.mod.ufmg.simulnormal.janela, id)
  
  prev.mod.ufmg.simulnormal.rodada <- read_csv("prev.mod.ufmg.simulnormal.rodada.csv",
                                               col_types = cols(pvm = col_number(), 
                                                                pe = col_number(), 
                                                                pvv = col_number()),
                                               col_names = T)
  prev.mod.ufmg.simulnormal.rodada <- arrange(prev.mod.ufmg.simulnormal.rodada, id)
  
  #####
  prev.mod.arr.SD0.geral <- read_csv("prev.mod.arr.SD0.geral.csv",
                                     col_types = cols(pvm = col_number(), 
                                                      pe = col_number(), 
                                                      pvv = col_number(),
                                                      golman = col_number(), 
                                                      golvis = col_number()),
                                     col_names = T)
  prev.mod.arr.SD0.geral <- arrange(prev.mod.arr.SD0.geral, id)
  
  prev.mod.arr.SD0.ano <- read_csv("prev.mod.arr.SD0.ano.csv",
                                   col_types = cols(pvm = col_number(), 
                                                    pe = col_number(), 
                                                    pvv = col_number(),
                                                    golman = col_number(), 
                                                    golvis = col_number()),
                                   col_names = T)
  prev.mod.arr.SD0.ano <- arrange(prev.mod.arr.SD0.ano, id)
  
  prev.mod.arr.SD0.janela <- read_csv("prev.mod.arr.SD0.janela.csv",
                                      col_types = cols(pvm = col_number(), 
                                                       pe = col_number(), 
                                                       pvv = col_number(),
                                                       golman = col_number(), 
                                                       golvis = col_number()),
                                      col_names = T)
  prev.mod.arr.SD0.janela <- arrange(prev.mod.arr.SD0.janela, id)
  
  prev.mod.arr.SD0.rodada <- read_csv("prev.mod.arr.SD0.rodada.csv",
                                      col_types = cols(pvm = col_number(), 
                                                       pe = col_number(), 
                                                       pvv = col_number(),
                                                       golman = col_number(), 
                                                       golvis = col_number()),
                                      col_names = T)
  prev.mod.arr.SD0.rodada <- arrange(prev.mod.arr.SD0.rodada, id)

  #####
  prev.mod.arr.SD1.geral <- read_csv("prev.mod.arr.SD1.geral.csv",
                                     col_types = cols(pvm = col_number(), 
                                                      pe = col_number(), 
                                                      pvv = col_number(),
                                                      golman = col_number(), 
                                                      golvis = col_number()),
                                     col_names = T)
  prev.mod.arr.SD1.geral <- arrange(prev.mod.arr.SD1.geral, id)
  
  prev.mod.arr.SD1.ano <- read_csv("prev.mod.arr.SD1.ano.csv",
                                   col_types = cols(pvm = col_number(), 
                                                    pe = col_number(), 
                                                    pvv = col_number(),
                                                    golman = col_number(), 
                                                    golvis = col_number()),
                                   col_names = T)
  prev.mod.arr.SD1.ano <- arrange(prev.mod.arr.SD1.ano, id)
  
  prev.mod.arr.SD1.janela <- read_csv("prev.mod.arr.SD1.janela.csv",
                                      col_types = cols(pvm = col_number(), 
                                                       pe = col_number(), 
                                                       pvv = col_number(),
                                                       golman = col_number(), 
                                                       golvis = col_number()),
                                      col_names = T)
  prev.mod.arr.SD1.janela <- arrange(prev.mod.arr.SD1.janela, id)
  
  prev.mod.arr.SD1.rodada <- read_csv("prev.mod.arr.SD1.rodada.csv",
                                      col_types = cols(pvm = col_number(), 
                                                       pe = col_number(), 
                                                       pvv = col_number(),
                                                       golman = col_number(), 
                                                       golvis = col_number()),
                                      col_names = T)
  prev.mod.arr.SD1.rodada <- arrange(prev.mod.arr.SD1.rodada, id)
  
  #####
  prev.mod.arr.chance1.geral <- read_csv("prev.mod.arr.chance1.geral.csv",
                                         col_types = cols(pvm = col_number(), 
                                                          pe = col_number(), 
                                                          pvv = col_number(),
                                                          golman = col_number(), 
                                                          golvis = col_number()),
                                         col_names = T)
  prev.mod.arr.chance1.geral <- arrange(prev.mod.arr.chance1.geral, id)
  
  prev.mod.arr.chance1.ano <- read_csv("prev.mod.arr.chance1.ano.csv",
                                       col_types = cols(pvm = col_number(), 
                                                        pe = col_number(), 
                                                        pvv = col_number(),
                                                        golman = col_number(), 
                                                        golvis = col_number()),
                                       col_names = T)
  prev.mod.arr.chance1.ano <- arrange(prev.mod.arr.chance1.ano, id)
  
  prev.mod.arr.chance1.janela <- read_csv("prev.mod.arr.chance1.janela.csv",
                                          col_types = cols(pvm = col_number(), 
                                                           pe = col_number(), 
                                                           pvv = col_number(),
                                                           golman = col_number(), 
                                                           golvis = col_number()),
                                          col_names = T)
  prev.mod.arr.chance1.janela <- arrange(prev.mod.arr.chance1.janela, id)
  
  prev.mod.arr.chance1.rodada <- read_csv("prev.mod.arr.chance1.rodada.csv",
                                          col_types = cols(pvm = col_number(), 
                                                           pe = col_number(), 
                                                           pvv = col_number(),
                                                           golman = col_number(), 
                                                           golvis = col_number()),
                                          col_names = T)
  prev.mod.arr.chance1.rodada <- arrange(prev.mod.arr.chance1.rodada, id)
  
  #####
  prev.mod.arr.chance2.geral <- read_csv("prev.mod.arr.chance2.geral.csv",
                                         col_types = cols(pvm = col_number(), 
                                                          pe = col_number(), 
                                                          pvv = col_number(),
                                                          golman = col_number(), 
                                                          golvis = col_number()),
                                         col_names = T)
  prev.mod.arr.chance2.geral <- arrange(prev.mod.arr.chance2.geral, id)
  
  prev.mod.arr.chance2.ano <- read_csv("prev.mod.arr.chance2.ano.csv",
                                       col_types = cols(pvm = col_number(), 
                                                        pe = col_number(), 
                                                        pvv = col_number(),
                                                        golman = col_number(), 
                                                        golvis = col_number()),
                                       col_names = T)
  prev.mod.arr.chance2.ano <- arrange(prev.mod.arr.chance2.ano, id)
  
  prev.mod.arr.chance2.janela <- read_csv("prev.mod.arr.chance2.janela.csv",
                                          col_types = cols(pvm = col_number(), 
                                                           pe = col_number(), 
                                                           pvv = col_number(),
                                                           golman = col_number(), 
                                                           golvis = col_number()),
                                          col_names = T)
  prev.mod.arr.chance2.janela <- arrange(prev.mod.arr.chance2.janela, id)
  
  prev.mod.arr.chance2.rodada <- read_csv("prev.mod.arr.chance2.rodada.csv",
                                          col_types = cols(pvm = col_number(), 
                                                           pe = col_number(), 
                                                           pvv = col_number(),
                                                           golman = col_number(), 
                                                           golvis = col_number()),
                                          col_names = T)
  prev.mod.arr.chance2.rodada <- arrange(prev.mod.arr.chance2.rodada, id)
}
# tabela de metricas
{
  # Listar os modelos
{
  #modelos
  {modelos <- c(
    "ConImp",     # mod.controle.implicito.media
    "ConTeiMan",  # mod.controle.mand
    "ConTeiEmp",  # mod.controle.emp
    "ConTeiVis",  # mod.controle.vist
    "ConMed",     # mod.controle.media
    "ConPois",    # mod.controle.poisson
    "ConUnif",    # mod.controle.uniforme
    
    "Ch1Ano",     # mod.arr.chance1.ano
    "Ch1Ger",     # mod.arr.chance1.geral
    "Ch1Sem",     # mod.arr.chance1.janela
    "Ch1Rod",     # mod.arr.chance1.rodada
    
    "Ch2Ano",     # mod.arr.chance2.ano
    "Ch2Ger",     # mod.arr.chance2.geral
    "Ch2Sem",     # mod.arr.chance2.janela
    "Ch2Rod",     # mod.arr.chance2.rodada
    
    "Sd0Ano",     # mod.arr.SD0.ano
    "Sd0Ger",     # mod.arr.SD0.geral
    "Sd0Sem",     # mod.arr.SD0.janela
    "Sd0Rod",     # mod.arr.SD0.rodada
    
    "Sd1Ano",     # mod.arr.SD1.ano
    "Sd1Ger",     # mod.arr.SD1.geral
    "Sd1Sem",     # mod.arr.SD1.janela
    "Sd1Rod",     # mod.arr.SD1.rodada
    
    "UfmgAno",    # mod.ufmg.comsimul.ano
    "UfmgGer",    # mod.ufmg.comsimul.geral
    "UfmgSem",    # mod.ufmg.comsimul.janela
    "UfmgRod",    # mod.ufmg.comsimul.rodada
    
    "UfmgSsAno",  # mod.ufmg.semsimul.ano
    "UfmgSsGer",  # mod.ufmg.semsimul.geral
    "UfmgSsSem",  # mod.ufmg.semsimul.janela
    "UfmgSsRod",  # mod.ufmg.semsimul.rodada
    
    "UfmgSnAno",  # mod.ufmg.simulnormal.ano
    "UfmgSnGer",  # mod.ufmg.simulnormal.geral
    "UfmgSnSem",  # mod.ufmg.simulnormal.janela
    "UfmgSnRod"   # mod.ufmg.simulnormal.rodada
  )}
  
  #definetti
  {definetti <- c(
    met.definetti(prev.mod.controle.implicito.media$pvm, prev.mod.controle.implicito.media$pe, prev.mod.controle.implicito.media$pvv, tab$resul, F),
    met.definetti(prev.mod.controle.mand$pvm, prev.mod.controle.mand$pe, prev.mod.controle.mand$pvv, tab$resul, F),
    met.definetti(prev.mod.controle.emp$pvm, prev.mod.controle.emp$pe, prev.mod.controle.emp$pvv, tab$resul, F),
    met.definetti(prev.mod.controle.vist$pvm, prev.mod.controle.vist$pe, prev.mod.controle.vist$pvv, tab$resul, F),
    "Não se aplica",
    "Não se aplica",
    met.definetti(prev.mod.controle.uniforme$pvm, prev.mod.controle.uniforme$pe, prev.mod.controle.uniforme$pvv, tab$resul, F),
    
    met.definetti(prev.mod.arr.chance1.ano$pvm, prev.mod.arr.chance1.ano$pe, prev.mod.arr.chance1.ano$pvv, tab$resul, F),
    met.definetti(prev.mod.arr.chance1.geral$pvm, prev.mod.arr.chance1.geral$pe, prev.mod.arr.chance1.geral$pvv, tab$resul, F),
    met.definetti(prev.mod.arr.chance1.janela$pvm, prev.mod.arr.chance1.janela$pe, prev.mod.arr.chance1.janela$pvv, tab$resul, F),
    met.definetti(prev.mod.arr.chance1.rodada$pvm, prev.mod.arr.chance1.rodada$pe, prev.mod.arr.chance1.rodada$pvv, tab$resul, F),
    
    met.definetti(prev.mod.arr.chance2.ano$pvm, prev.mod.arr.chance2.ano$pe, prev.mod.arr.chance2.ano$pvv, tab$resul, F),
    met.definetti(prev.mod.arr.chance2.geral$pvm, prev.mod.arr.chance2.geral$pe, prev.mod.arr.chance2.geral$pvv, tab$resul, F),
    met.definetti(prev.mod.arr.chance2.janela$pvm, prev.mod.arr.chance2.janela$pe, prev.mod.arr.chance2.janela$pvv, tab$resul, F),
    met.definetti(prev.mod.arr.chance2.rodada$pvm, prev.mod.arr.chance2.rodada$pe, prev.mod.arr.chance2.rodada$pvv, tab$resul, F),
    
    met.definetti(prev.mod.arr.SD0.ano$pvm, prev.mod.arr.SD0.ano$pe, prev.mod.arr.SD0.ano$pvv, tab$resul, F),
    met.definetti(prev.mod.arr.SD0.geral$pvm, prev.mod.arr.SD0.geral$pe, prev.mod.arr.SD0.geral$pvv, tab$resul, F),
    met.definetti(prev.mod.arr.SD0.janela$pvm, prev.mod.arr.SD0.janela$pe, prev.mod.arr.SD0.janela$pvv, tab$resul, F),
    met.definetti(prev.mod.arr.SD0.rodada$pvm, prev.mod.arr.SD0.rodada$pe, prev.mod.arr.SD0.rodada$pvv, tab$resul, F),
    
    met.definetti(prev.mod.arr.SD1.ano$pvm, prev.mod.arr.SD1.ano$pe, prev.mod.arr.SD1.ano$pvv, tab$resul, F),
    met.definetti(prev.mod.arr.SD1.geral$pvm, prev.mod.arr.SD1.geral$pe, prev.mod.arr.SD1.geral$pvv, tab$resul, F),
    met.definetti(prev.mod.arr.SD1.janela$pvm, prev.mod.arr.SD1.janela$pe, prev.mod.arr.SD1.janela$pvv, tab$resul, F),
    met.definetti(prev.mod.arr.SD1.rodada$pvm, prev.mod.arr.SD1.rodada$pe, prev.mod.arr.SD1.rodada$pvv, tab$resul, F),
    
    met.definetti(prev.mod.ufmg.comsimul.ano$pvm, prev.mod.ufmg.comsimul.ano$pe, prev.mod.ufmg.comsimul.ano$pvv, tab$resul, F),
    met.definetti(prev.mod.ufmg.comsimul.geral$pvm, prev.mod.ufmg.comsimul.geral$pe, prev.mod.ufmg.comsimul.geral$pvv, tab$resul, F),
    met.definetti(prev.mod.ufmg.comsimul.janela$pvm, prev.mod.ufmg.comsimul.janela$pe, prev.mod.ufmg.comsimul.janela$pvv, tab$resul, F),
    met.definetti(prev.mod.ufmg.comsimul.rodada$pvm, prev.mod.ufmg.comsimul.rodada$pe, prev.mod.ufmg.comsimul.rodada$pvv, tab$resul, F),
    
    met.definetti(prev.mod.ufmg.semsimul.ano$pvm, prev.mod.ufmg.semsimul.ano$pe, prev.mod.ufmg.semsimul.ano$pvv, tab$resul, F),
    met.definetti(prev.mod.ufmg.semsimul.geral$pvm, prev.mod.ufmg.semsimul.geral$pe, prev.mod.ufmg.semsimul.geral$pvv, tab$resul, F),
    met.definetti(prev.mod.ufmg.semsimul.janela$pvm, prev.mod.ufmg.semsimul.janela$pe, prev.mod.ufmg.semsimul.janela$pvv, tab$resul, F),
    met.definetti(prev.mod.ufmg.semsimul.rodada$pvm, prev.mod.ufmg.semsimul.rodada$pe, prev.mod.ufmg.semsimul.rodada$pvv, tab$resul, F),
    
    met.definetti(prev.mod.ufmg.simulnormal.ano$pvm, prev.mod.ufmg.simulnormal.ano$pe, prev.mod.ufmg.simulnormal.ano$pvv, tab$resul, F),
    met.definetti(prev.mod.ufmg.simulnormal.geral$pvm, prev.mod.ufmg.simulnormal.geral$pe, prev.mod.ufmg.simulnormal.geral$pvv, tab$resul, F),
    met.definetti(prev.mod.ufmg.simulnormal.janela$pvm, prev.mod.ufmg.simulnormal.janela$pe, prev.mod.ufmg.simulnormal.janela$pvv, tab$resul, F),
    met.definetti(prev.mod.ufmg.simulnormal.rodada$pvm, prev.mod.ufmg.simulnormal.rodada$pe, prev.mod.ufmg.simulnormal.rodada$pvv, tab$resul, F)
  )}
  
  # definetti detalhada
  {definetti.detalhada <- c(
    met.definetti.detalhada(tab$plac_mand, tab$plac_vist, prev.mod.controle.implicito.media$golman, prev.mod.controle.implicito.media$golvis, F),
    "Não se aplica",
    "Não se aplica",
    "Não se aplica",
    met.definetti.detalhada(tab$plac_mand, tab$plac_vist, prev.mod.controle.media$golman, prev.mod.controle.media$golvis, F),
    met.definetti.detalhada(tab$plac_mand, tab$plac_vist, prev.mod.controle.poisson$golman, prev.mod.controle.poisson$golvis, F),
    "Não se aplica",
    
    met.definetti.detalhada(tab$plac_mand, tab$plac_vist, prev.mod.arr.chance1.ano$golman, prev.mod.arr.chance1.ano$golvis, F),
    met.definetti.detalhada(tab$plac_mand, tab$plac_vist, prev.mod.arr.chance1.geral$golman, prev.mod.arr.chance1.geral$golvis, F),
    met.definetti.detalhada(tab$plac_mand, tab$plac_vist, prev.mod.arr.chance1.janela$golman, prev.mod.arr.chance1.janela$golvis, F),
    met.definetti.detalhada(tab$plac_mand, tab$plac_vist, prev.mod.arr.chance1.rodada$golman, prev.mod.arr.chance1.rodada$golvis, F),
    
    met.definetti.detalhada(tab$plac_mand, tab$plac_vist, prev.mod.arr.chance2.ano$golman, prev.mod.arr.chance2.ano$golvis, F),
    met.definetti.detalhada(tab$plac_mand, tab$plac_vist, prev.mod.arr.chance2.geral$golman, prev.mod.arr.chance2.geral$golvis, F),
    met.definetti.detalhada(tab$plac_mand, tab$plac_vist, prev.mod.arr.chance2.janela$golman, prev.mod.arr.chance2.janela$golvis, F),
    met.definetti.detalhada(tab$plac_mand, tab$plac_vist, prev.mod.arr.chance2.rodada$golman, prev.mod.arr.chance2.rodada$golvis, F),
    
    met.definetti.detalhada(tab$plac_mand, tab$plac_vist, prev.mod.arr.SD0.ano$golman, prev.mod.arr.SD0.ano$golvis, F),
    met.definetti.detalhada(tab$plac_mand, tab$plac_vist, prev.mod.arr.SD0.geral$golman, prev.mod.arr.SD0.geral$golvis, F),
    met.definetti.detalhada(tab$plac_mand, tab$plac_vist, prev.mod.arr.SD0.janela$golman, prev.mod.arr.SD0.janela$golvis, F),
    met.definetti.detalhada(tab$plac_mand, tab$plac_vist, prev.mod.arr.SD0.rodada$golman, prev.mod.arr.SD0.rodada$golvis, F),
    
    met.definetti.detalhada(tab$plac_mand, tab$plac_vist, prev.mod.arr.SD1.ano$golman, prev.mod.arr.SD1.ano$golvis, F),
    met.definetti.detalhada(tab$plac_mand, tab$plac_vist, prev.mod.arr.SD1.geral$golman, prev.mod.arr.SD1.geral$golvis, F),
    met.definetti.detalhada(tab$plac_mand, tab$plac_vist, prev.mod.arr.SD1.janela$golman, prev.mod.arr.SD1.janela$golvis, F),
    met.definetti.detalhada(tab$plac_mand, tab$plac_vist, prev.mod.arr.SD1.rodada$golman, prev.mod.arr.SD1.rodada$golvis, F),
    
    "Não se aplica",
    "Não se aplica",
    "Não se aplica",
    "Não se aplica",
    
    "Não se aplica",
    "Não se aplica",
    "Não se aplica",
    "Não se aplica",
    
    "Não se aplica",
    "Não se aplica",
    "Não se aplica",
    "Não se aplica"
  )}
  
  #taxa de acerto
  {taxa_acerto <- c(
    met.taxa.acerto(prev.mod.controle.implicito.media$resultado, tab$resul, FALSE),
    met.taxa.acerto(prev.mod.controle.mand$resultado, tab$resul, FALSE),
    met.taxa.acerto(prev.mod.controle.emp$resultado, tab$resul, FALSE),
    met.taxa.acerto(prev.mod.controle.vist$resultado, tab$resul, FALSE),
    met.taxa.acerto(prev.mod.controle.media$resultado, tab$resul, FALSE),
    met.taxa.acerto(prev.mod.controle.poisson$resultado, tab$resul, FALSE),
    met.taxa.acerto(prev.mod.controle.uniforme$resultado, tab$resul, FALSE),
    
    met.taxa.acerto(prev.mod.arr.chance1.ano$resultado, tab$resul, FALSE),
    met.taxa.acerto(prev.mod.arr.chance1.geral$resultado, tab$resul, FALSE),
    met.taxa.acerto(prev.mod.arr.chance1.janela$resultado, tab$resul, FALSE),
    met.taxa.acerto(prev.mod.arr.chance1.rodada$resultado, tab$resul, FALSE),
    
    met.taxa.acerto(prev.mod.arr.chance2.ano$resultado, tab$resul, FALSE),
    met.taxa.acerto(prev.mod.arr.chance2.geral$resultado, tab$resul, FALSE),
    met.taxa.acerto(prev.mod.arr.chance2.janela$resultado, tab$resul, FALSE),
    met.taxa.acerto(prev.mod.arr.chance2.rodada$resultado, tab$resul, FALSE),
    
    met.taxa.acerto(prev.mod.arr.SD0.ano$resultado, tab$resul, FALSE),
    met.taxa.acerto(prev.mod.arr.SD0.geral$resultado, tab$resul, FALSE),
    met.taxa.acerto(prev.mod.arr.SD0.janela$resultado, tab$resul, FALSE),
    met.taxa.acerto(prev.mod.arr.SD0.rodada$resultado, tab$resul, FALSE),
    
    met.taxa.acerto(prev.mod.arr.SD1.ano$resultado, tab$resul, FALSE),
    met.taxa.acerto(prev.mod.arr.SD1.geral$resultado, tab$resul, FALSE),
    met.taxa.acerto(prev.mod.arr.SD1.janela$resultado, tab$resul, FALSE),
    met.taxa.acerto(prev.mod.arr.SD1.rodada$resultado, tab$resul, FALSE),
    
    met.taxa.acerto(prev.mod.ufmg.comsimul.ano$resultado, tab$resul, FALSE),
    met.taxa.acerto(prev.mod.ufmg.comsimul.geral$resultado, tab$resul, FALSE),
    met.taxa.acerto(prev.mod.ufmg.comsimul.janela$resultado, tab$resul, FALSE),
    met.taxa.acerto(prev.mod.ufmg.comsimul.rodada$resultado, tab$resul, FALSE),
    
    met.taxa.acerto(prev.mod.ufmg.semsimul.ano$resultado, tab$resul, FALSE),
    met.taxa.acerto(prev.mod.ufmg.semsimul.geral$resultado, tab$resul, FALSE),
    met.taxa.acerto(prev.mod.ufmg.semsimul.janela$resultado, tab$resul, FALSE),
    met.taxa.acerto(prev.mod.ufmg.semsimul.rodada$resultado, tab$resul, FALSE),
    
    met.taxa.acerto(prev.mod.ufmg.simulnormal.ano$resultado, tab$resul, FALSE),
    met.taxa.acerto(prev.mod.ufmg.simulnormal.geral$resultado, tab$resul, FALSE),
    met.taxa.acerto(prev.mod.ufmg.simulnormal.janela$resultado, tab$resul, FALSE),
    met.taxa.acerto(prev.mod.ufmg.simulnormal.rodada$resultado, tab$resul, FALSE)
  )}
  
  #epmp
  {epmp <- c(
    met.epmp(prev.mod.controle.implicito.media$resultado, tab$resul, prev.mod.controle.implicito.media$pvm, prev.mod.controle.implicito.media$pe, prev.mod.controle.implicito.media$pvv, FALSE),
    met.epmp(prev.mod.controle.mand$resultado, tab$resul, prev.mod.controle.mand$pvm, prev.mod.controle.mand$pe, prev.mod.controle.mand$pvv, FALSE),
    met.epmp(prev.mod.controle.emp$resultado, tab$resul, prev.mod.controle.emp$pvm, prev.mod.controle.emp$pe, prev.mod.controle.emp$pvv, FALSE),
    met.epmp(prev.mod.controle.vist$resultado, tab$resul, prev.mod.controle.vist$pvm, prev.mod.controle.vist$pe, prev.mod.controle.vist$pvv, FALSE),
    "Não se aplica",
    "Não se aplica",
    met.epmp(prev.mod.controle.uniforme$resultado, tab$resul, prev.mod.controle.uniforme$pvm, prev.mod.controle.uniforme$pe, prev.mod.controle.uniforme$pvv, FALSE),
    
    met.epmp(prev.mod.arr.chance1.ano$resultado, tab$resul, prev.mod.arr.chance1.ano$pvm, prev.mod.arr.chance1.ano$pe, prev.mod.arr.chance1.ano$pvv, FALSE),
    met.epmp(prev.mod.arr.chance1.geral$resultado, tab$resul, prev.mod.arr.chance1.geral$pvm, prev.mod.arr.chance1.geral$pe, prev.mod.arr.chance1.geral$pvv, FALSE),
    met.epmp(prev.mod.arr.chance1.janela$resultado, tab$resul, prev.mod.arr.chance1.janela$pvm, prev.mod.arr.chance1.janela$pe, prev.mod.arr.chance1.janela$pvv, FALSE),
    met.epmp(prev.mod.arr.chance1.rodada$resultado, tab$resul, prev.mod.arr.chance1.rodada$pvm, prev.mod.arr.chance1.rodada$pe, prev.mod.arr.chance1.rodada$pvv, FALSE),
    
    met.epmp(prev.mod.arr.chance2.ano$resultado, tab$resul, prev.mod.arr.chance2.ano$pvm, prev.mod.arr.chance2.ano$pe, prev.mod.arr.chance2.ano$pvv, FALSE),
    met.epmp(prev.mod.arr.chance2.geral$resultado, tab$resul, prev.mod.arr.chance2.geral$pvm, prev.mod.arr.chance2.geral$pe, prev.mod.arr.chance2.geral$pvv, FALSE),
    met.epmp(prev.mod.arr.chance2.janela$resultado, tab$resul, prev.mod.arr.chance2.janela$pvm, prev.mod.arr.chance2.janela$pe, prev.mod.arr.chance2.janela$pvv, FALSE),
    met.epmp(prev.mod.arr.chance2.rodada$resultado, tab$resul, prev.mod.arr.chance2.rodada$pvm, prev.mod.arr.chance2.rodada$pe, prev.mod.arr.chance2.rodada$pvv, FALSE),
    
    met.epmp(prev.mod.arr.SD0.ano$resultado, tab$resul, prev.mod.arr.SD0.ano$pvm, prev.mod.arr.SD0.ano$pe, prev.mod.arr.SD0.ano$pvv, FALSE),
    met.epmp(prev.mod.arr.SD0.geral$resultado, tab$resul, prev.mod.arr.SD0.geral$pvm, prev.mod.arr.SD0.geral$pe, prev.mod.arr.SD0.geral$pvv, FALSE),
    met.epmp(prev.mod.arr.SD0.janela$resultado, tab$resul, prev.mod.arr.SD0.janela$pvm, prev.mod.arr.SD0.janela$pe, prev.mod.arr.SD0.janela$pvv, FALSE),
    met.epmp(prev.mod.arr.SD0.rodada$resultado, tab$resul, prev.mod.arr.SD0.rodada$pvm, prev.mod.arr.SD0.rodada$pe, prev.mod.arr.SD0.rodada$pvv, FALSE),
    
    met.epmp(prev.mod.arr.SD1.ano$resultado, tab$resul, prev.mod.arr.SD1.ano$pvm, prev.mod.arr.SD1.ano$pe, prev.mod.arr.SD1.ano$pvv, FALSE),
    met.epmp(prev.mod.arr.SD1.geral$resultado, tab$resul, prev.mod.arr.SD1.geral$pvm, prev.mod.arr.SD1.geral$pe, prev.mod.arr.SD1.geral$pvv, FALSE),
    met.epmp(prev.mod.arr.SD1.janela$resultado, tab$resul, prev.mod.arr.SD1.janela$pvm, prev.mod.arr.SD1.janela$pe, prev.mod.arr.SD1.janela$pvv, FALSE),
    met.epmp(prev.mod.arr.SD1.rodada$resultado, tab$resul, prev.mod.arr.SD1.rodada$pvm, prev.mod.arr.SD1.rodada$pe, prev.mod.arr.SD1.rodada$pvv, FALSE),
    
    met.epmp(prev.mod.ufmg.comsimul.ano$resultado, tab$resul, prev.mod.ufmg.comsimul.ano$pvm, prev.mod.ufmg.comsimul.ano$pe, prev.mod.ufmg.comsimul.ano$pvv, FALSE),
    met.epmp(prev.mod.ufmg.comsimul.geral$resultado, tab$resul, prev.mod.ufmg.comsimul.geral$pvm, prev.mod.ufmg.comsimul.geral$pe, prev.mod.ufmg.comsimul.geral$pvv, FALSE),
    met.epmp(prev.mod.ufmg.comsimul.janela$resultado, tab$resul, prev.mod.ufmg.comsimul.janela$pvm, prev.mod.ufmg.comsimul.janela$pe, prev.mod.ufmg.comsimul.janela$pvv, FALSE),
    met.epmp(prev.mod.ufmg.comsimul.rodada$resultado, tab$resul, prev.mod.ufmg.comsimul.rodada$pvm, prev.mod.ufmg.comsimul.rodada$pe, prev.mod.ufmg.comsimul.rodada$pvv, FALSE),
    
    met.epmp(prev.mod.ufmg.semsimul.ano$resultado, tab$resul, prev.mod.ufmg.semsimul.ano$pvm, prev.mod.ufmg.semsimul.ano$pe, prev.mod.ufmg.semsimul.ano$pvv, FALSE),
    met.epmp(prev.mod.ufmg.semsimul.geral$resultado, tab$resul, prev.mod.ufmg.semsimul.geral$pvm, prev.mod.ufmg.semsimul.geral$pe, prev.mod.ufmg.semsimul.geral$pvv, FALSE),
    met.epmp(prev.mod.ufmg.semsimul.janela$resultado, tab$resul, prev.mod.ufmg.semsimul.janela$pvm, prev.mod.ufmg.semsimul.janela$pe, prev.mod.ufmg.semsimul.janela$pvv, FALSE),
    met.epmp(prev.mod.ufmg.semsimul.rodada$resultado, tab$resul, prev.mod.ufmg.semsimul.rodada$pvm, prev.mod.ufmg.semsimul.rodada$pe, prev.mod.ufmg.semsimul.rodada$pvv, FALSE),
    
    met.epmp(prev.mod.ufmg.simulnormal.ano$resultado, tab$resul, prev.mod.ufmg.simulnormal.ano$pvm, prev.mod.ufmg.simulnormal.ano$pe, prev.mod.ufmg.simulnormal.ano$pvv, FALSE),
    met.epmp(prev.mod.ufmg.simulnormal.geral$resultado, tab$resul, prev.mod.ufmg.simulnormal.geral$pvm, prev.mod.ufmg.simulnormal.geral$pe, prev.mod.ufmg.simulnormal.geral$pvv, FALSE),
    met.epmp(prev.mod.ufmg.simulnormal.janela$resultado, tab$resul, prev.mod.ufmg.simulnormal.janela$pvm, prev.mod.ufmg.simulnormal.janela$pe, prev.mod.ufmg.simulnormal.janela$pvv, FALSE),
    met.epmp(prev.mod.ufmg.simulnormal.rodada$resultado, tab$resul, prev.mod.ufmg.simulnormal.rodada$pvm, prev.mod.ufmg.simulnormal.rodada$pe, prev.mod.ufmg.simulnormal.rodada$pvv, FALSE)
  )}
  
  #taxa de funcionamento
  {taxa_func <- c(
    met.taxa.func(prev.mod.controle.implicito.media$resultado),
    met.taxa.func(prev.mod.controle.mand$resultado),
    met.taxa.func(prev.mod.controle.emp$resultado),
    met.taxa.func(prev.mod.controle.vist$resultado),
    met.taxa.func(prev.mod.controle.media$resultado),
    met.taxa.func(prev.mod.controle.poisson$resultado),
    met.taxa.func(prev.mod.controle.uniforme$resultado),
    
    met.taxa.func(prev.mod.arr.chance1.ano$resultado),
    met.taxa.func(prev.mod.arr.chance1.geral$resultado),
    met.taxa.func(prev.mod.arr.chance1.janela$resultado),
    met.taxa.func(prev.mod.arr.chance1.rodada$resultado),
    
    met.taxa.func(prev.mod.arr.chance2.ano$resultado),
    met.taxa.func(prev.mod.arr.chance2.geral$resultado),
    met.taxa.func(prev.mod.arr.chance2.janela$resultado),
    met.taxa.func(prev.mod.arr.chance2.rodada$resultado),
    
    met.taxa.func(prev.mod.arr.SD0.ano$resultado),
    met.taxa.func(prev.mod.arr.SD0.geral$resultado),
    met.taxa.func(prev.mod.arr.SD0.janela$resultado),
    met.taxa.func(prev.mod.arr.SD0.rodada$resultado),
    
    met.taxa.func(prev.mod.arr.SD1.ano$resultado),
    met.taxa.func(prev.mod.arr.SD1.geral$resultado),
    met.taxa.func(prev.mod.arr.SD1.janela$resultado),
    met.taxa.func(prev.mod.arr.SD1.rodada$resultado),
    
    met.taxa.func(prev.mod.ufmg.comsimul.ano$resultado),
    met.taxa.func(prev.mod.ufmg.comsimul.geral$resultado),
    met.taxa.func(prev.mod.ufmg.comsimul.janela$resultado),
    met.taxa.func(prev.mod.ufmg.comsimul.rodada$resultado),
    
    met.taxa.func(prev.mod.ufmg.semsimul.ano$resultado),
    met.taxa.func(prev.mod.ufmg.semsimul.geral$resultado),
    met.taxa.func(prev.mod.ufmg.semsimul.janela$resultado),
    met.taxa.func(prev.mod.ufmg.semsimul.rodada$resultado),
    
    met.taxa.func(prev.mod.ufmg.simulnormal.ano$resultado),
    met.taxa.func(prev.mod.ufmg.simulnormal.geral$resultado),
    met.taxa.func(prev.mod.ufmg.simulnormal.janela$resultado),
    met.taxa.func(prev.mod.ufmg.simulnormal.rodada$resultado)
  )}
  
  #nível de complexidade
  {tab.nc <- read_csv("~/Documents/Dissertação/scripts/bancos/tab.complexidade.csv", 
                                locale = locale(decimal_mark = ","))
    Complexidade <- tab.nc$NC
  }
  
  #matriz de confusão - precision
  {mc_precision <- c(
    met.cm(prev.mod.controle.implicito.media$resultado, tab$resul, FALSE)$Precision,
    "Não se aplica",
    "Não se aplica",
    "Não se aplica",
    met.cm(prev.mod.controle.media$resultado, tab$resul, FALSE)$Precision,
    met.cm(prev.mod.controle.poisson$resultado, tab$resul, FALSE)$Precision,
    met.cm(prev.mod.controle.uniforme$resultado, tab$resul, FALSE)$Precision,
    
    met.cm(prev.mod.arr.chance1.ano$resultado, tab$resul, FALSE)$Precision,
    met.cm(prev.mod.arr.chance1.geral$resultado, tab$resul, FALSE)$Precision,
    met.cm(prev.mod.arr.chance1.janela$resultado, tab$resul, FALSE)$Precision,
    met.cm(prev.mod.arr.chance1.rodada$resultado, tab$resul, FALSE)$Precision,
    
    met.cm(prev.mod.arr.chance2.ano$resultado, tab$resul, FALSE)$Precision,
    met.cm(prev.mod.arr.chance2.geral$resultado, tab$resul, FALSE)$Precision,
    met.cm(prev.mod.arr.chance2.janela$resultado, tab$resul, FALSE)$Precision,
    met.cm(prev.mod.arr.chance2.rodada$resultado, tab$resul, FALSE)$Precision,
    
    met.cm(prev.mod.arr.SD0.ano$resultado, tab$resul, FALSE)$Precision,
    met.cm(prev.mod.arr.SD0.geral$resultado, tab$resul, FALSE)$Precision,
    met.cm(prev.mod.arr.SD0.janela$resultado, tab$resul, FALSE)$Precision,
    met.cm(prev.mod.arr.SD0.rodada$resultado, tab$resul, FALSE)$Precision,
    
    met.cm(prev.mod.arr.SD1.ano$resultado, tab$resul, FALSE)$Precision,
    met.cm(prev.mod.arr.SD1.geral$resultado, tab$resul, FALSE)$Precision,
    met.cm(prev.mod.arr.SD1.janela$resultado, tab$resul, FALSE)$Precision,
    met.cm(prev.mod.arr.SD1.rodada$resultado, tab$resul, FALSE)$Precision,
    
    met.cm(prev.mod.ufmg.comsimul.ano$resultado, tab$resul, FALSE)$Precision,
    met.cm(prev.mod.ufmg.comsimul.geral$resultado, tab$resul, FALSE)$Precision,#precisa adicionar pelo menos um empate para não dar erro de dimensionamento.
    met.cm(prev.mod.ufmg.comsimul.janela$resultado, tab$resul, FALSE)$Precision,
    met.cm(prev.mod.ufmg.comsimul.rodada$resultado, tab$resul, FALSE)$Precision,
    
    met.cm(prev.mod.ufmg.semsimul.ano$resultado, tab$resul, FALSE)$Precision,
    met.cm(prev.mod.ufmg.semsimul.geral$resultado, tab$resul, FALSE)$Precision,
    met.cm(prev.mod.ufmg.semsimul.janela$resultado, tab$resul, FALSE)$Precision,
    met.cm(prev.mod.ufmg.semsimul.rodada$resultado, tab$resul, FALSE)$Precision,
    
    met.cm(prev.mod.ufmg.simulnormal.ano$resultado, tab$resul, FALSE)$Precision,
    met.cm(prev.mod.ufmg.simulnormal.geral$resultado, tab$resul, FALSE)$Precision,
    met.cm(prev.mod.ufmg.simulnormal.janela$resultado, tab$resul, FALSE)$Precision,
    met.cm(prev.mod.ufmg.simulnormal.rodada$resultado, tab$resul, FALSE)$Precision
  )}
  
  #matriz de confusão - recall
  {mc_recall <- c(
    met.cm(prev.mod.controle.implicito.media$resultado, tab$resul, FALSE)$Recall,
    "Não se aplica",
    "Não se aplica",
    "Não se aplica",
    met.cm(prev.mod.controle.media$resultado, tab$resul, FALSE)$Recall,
    met.cm(prev.mod.controle.poisson$resultado, tab$resul, FALSE)$Recall,
    met.cm(prev.mod.controle.uniforme$resultado, tab$resul, FALSE)$Recall,
    
    met.cm(prev.mod.arr.chance1.ano$resultado, tab$resul, FALSE)$Recall,
    met.cm(prev.mod.arr.chance1.geral$resultado, tab$resul, FALSE)$Recall,
    met.cm(prev.mod.arr.chance1.janela$resultado, tab$resul, FALSE)$Recall,
    met.cm(prev.mod.arr.chance1.rodada$resultado, tab$resul, FALSE)$Recall,
    
    met.cm(prev.mod.arr.chance2.ano$resultado, tab$resul, FALSE)$Recall,
    met.cm(prev.mod.arr.chance2.geral$resultado, tab$resul, FALSE)$Recall,
    met.cm(prev.mod.arr.chance2.janela$resultado, tab$resul, FALSE)$Recall,
    met.cm(prev.mod.arr.chance2.rodada$resultado, tab$resul, FALSE)$Recall,
    
    met.cm(prev.mod.arr.SD0.ano$resultado, tab$resul, FALSE)$Recall,
    met.cm(prev.mod.arr.SD0.geral$resultado, tab$resul, FALSE)$Recall,
    met.cm(prev.mod.arr.SD0.janela$resultado, tab$resul, FALSE)$Recall,
    met.cm(prev.mod.arr.SD0.rodada$resultado, tab$resul, FALSE)$Recall,
    
    met.cm(prev.mod.arr.SD1.ano$resultado, tab$resul, FALSE)$Recall,
    met.cm(prev.mod.arr.SD1.geral$resultado, tab$resul, FALSE)$Recall,
    met.cm(prev.mod.arr.SD1.janela$resultado, tab$resul, FALSE)$Recall,
    met.cm(prev.mod.arr.SD1.rodada$resultado, tab$resul, FALSE)$Recall,
    
    met.cm(prev.mod.ufmg.comsimul.ano$resultado, tab$resul, FALSE)$Recall,
    met.cm(prev.mod.ufmg.comsimul.geral$resultado, tab$resul, FALSE)$Recall,#precisa adicionar pelo menos um empate para não dar erro de dimensionamento.
    met.cm(prev.mod.ufmg.comsimul.janela$resultado, tab$resul, FALSE)$Recall,
    met.cm(prev.mod.ufmg.comsimul.rodada$resultado, tab$resul, FALSE)$Recall,
    
    met.cm(prev.mod.ufmg.semsimul.ano$resultado, tab$resul, FALSE)$Recall,
    met.cm(prev.mod.ufmg.semsimul.geral$resultado, tab$resul, FALSE)$Recall,
    met.cm(prev.mod.ufmg.semsimul.janela$resultado, tab$resul, FALSE)$Recall,
    met.cm(prev.mod.ufmg.semsimul.rodada$resultado, tab$resul, FALSE)$Recall,
    
    met.cm(prev.mod.ufmg.simulnormal.ano$resultado, tab$resul, FALSE)$Recall,
    met.cm(prev.mod.ufmg.simulnormal.geral$resultado, tab$resul, FALSE)$Recall,
    met.cm(prev.mod.ufmg.simulnormal.janela$resultado, tab$resul, FALSE)$Recall,
    met.cm(prev.mod.ufmg.simulnormal.rodada$resultado, tab$resul, FALSE)$Recall
  )}
  
  #matriz de confusão - F1 Score
  {mc_f1 <- c(
    met.cm(prev.mod.controle.implicito.media$resultado, tab$resul, FALSE)$F1,
    "Não se aplica",
    "Não se aplica",
    "Não se aplica",
    met.cm(prev.mod.controle.media$resultado, tab$resul, FALSE)$F1,
    met.cm(prev.mod.controle.poisson$resultado, tab$resul, FALSE)$F1,
    met.cm(prev.mod.controle.uniforme$resultado, tab$resul, FALSE)$F1,
    
    met.cm(prev.mod.arr.chance1.ano$resultado, tab$resul, FALSE)$F1,
    met.cm(prev.mod.arr.chance1.geral$resultado, tab$resul, FALSE)$F1,
    met.cm(prev.mod.arr.chance1.janela$resultado, tab$resul, FALSE)$F1,
    met.cm(prev.mod.arr.chance1.rodada$resultado, tab$resul, FALSE)$F1,
    
    met.cm(prev.mod.arr.chance2.ano$resultado, tab$resul, FALSE)$F1,
    met.cm(prev.mod.arr.chance2.geral$resultado, tab$resul, FALSE)$F1,
    met.cm(prev.mod.arr.chance2.janela$resultado, tab$resul, FALSE)$F1,
    met.cm(prev.mod.arr.chance2.rodada$resultado, tab$resul, FALSE)$F1,
    
    met.cm(prev.mod.arr.SD0.ano$resultado, tab$resul, FALSE)$F1,
    met.cm(prev.mod.arr.SD0.geral$resultado, tab$resul, FALSE)$F1,
    met.cm(prev.mod.arr.SD0.janela$resultado, tab$resul, FALSE)$F1,
    met.cm(prev.mod.arr.SD0.rodada$resultado, tab$resul, FALSE)$F1,
    
    met.cm(prev.mod.arr.SD1.ano$resultado, tab$resul, FALSE)$F1,
    met.cm(prev.mod.arr.SD1.geral$resultado, tab$resul, FALSE)$F1,
    met.cm(prev.mod.arr.SD1.janela$resultado, tab$resul, FALSE)$F1,
    met.cm(prev.mod.arr.SD1.rodada$resultado, tab$resul, FALSE)$F1,
    
    met.cm(prev.mod.ufmg.comsimul.ano$resultado, tab$resul, FALSE)$F1,
    met.cm(prev.mod.ufmg.comsimul.geral$resultado, tab$resul, FALSE)$F1,#precisa adicionar pelo menos um empate para não dar erro de dimensionamento.
    met.cm(prev.mod.ufmg.comsimul.janela$resultado, tab$resul, FALSE)$F1,
    met.cm(prev.mod.ufmg.comsimul.rodada$resultado, tab$resul, FALSE)$F1,
    
    met.cm(prev.mod.ufmg.semsimul.ano$resultado, tab$resul, FALSE)$F1,
    met.cm(prev.mod.ufmg.semsimul.geral$resultado, tab$resul, FALSE)$F1,
    met.cm(prev.mod.ufmg.semsimul.janela$resultado, tab$resul, FALSE)$F1,
    met.cm(prev.mod.ufmg.semsimul.rodada$resultado, tab$resul, FALSE)$F1,
    
    met.cm(prev.mod.ufmg.simulnormal.ano$resultado, tab$resul, FALSE)$F1,
    met.cm(prev.mod.ufmg.simulnormal.geral$resultado, tab$resul, FALSE)$F1,
    met.cm(prev.mod.ufmg.simulnormal.janela$resultado, tab$resul, FALSE)$F1,
    met.cm(prev.mod.ufmg.simulnormal.rodada$resultado, tab$resul, FALSE)$F1
  )}
  
  #matriz de confusão - MCC
  {mc_MCC <- c(
    met.cm(prev.mod.controle.implicito.media$resultado, tab$resul, FALSE)$MCC,
    "Não se aplica",
    "Não se aplica",
    "Não se aplica",
    met.cm(prev.mod.controle.media$resultado, tab$resul, FALSE)$MCC,
    met.cm(prev.mod.controle.poisson$resultado, tab$resul, FALSE)$MCC,
    met.cm(prev.mod.controle.uniforme$resultado, tab$resul, FALSE)$MCC,
    
    met.cm(prev.mod.arr.chance1.ano$resultado, tab$resul, FALSE)$MCC,
    met.cm(prev.mod.arr.chance1.geral$resultado, tab$resul, FALSE)$MCC,
    met.cm(prev.mod.arr.chance1.janela$resultado, tab$resul, FALSE)$MCC,
    met.cm(prev.mod.arr.chance1.rodada$resultado, tab$resul, FALSE)$MCC,
    
    met.cm(prev.mod.arr.chance2.ano$resultado, tab$resul, FALSE)$MCC,
    met.cm(prev.mod.arr.chance2.geral$resultado, tab$resul, FALSE)$MCC,
    met.cm(prev.mod.arr.chance2.janela$resultado, tab$resul, FALSE)$MCC,
    met.cm(prev.mod.arr.chance2.rodada$resultado, tab$resul, FALSE)$MCC,
    
    met.cm(prev.mod.arr.SD0.ano$resultado, tab$resul, FALSE)$MCC,
    met.cm(prev.mod.arr.SD0.geral$resultado, tab$resul, FALSE)$MCC,
    met.cm(prev.mod.arr.SD0.janela$resultado, tab$resul, FALSE)$MCC,
    met.cm(prev.mod.arr.SD0.rodada$resultado, tab$resul, FALSE)$MCC,
    
    met.cm(prev.mod.arr.SD1.ano$resultado, tab$resul, FALSE)$MCC,
    met.cm(prev.mod.arr.SD1.geral$resultado, tab$resul, FALSE)$MCC,
    met.cm(prev.mod.arr.SD1.janela$resultado, tab$resul, FALSE)$MCC,
    met.cm(prev.mod.arr.SD1.rodada$resultado, tab$resul, FALSE)$MCC,
    
    met.cm(prev.mod.ufmg.comsimul.ano$resultado, tab$resul, FALSE)$MCC,
    met.cm(prev.mod.ufmg.comsimul.geral$resultado, tab$resul, FALSE)$MCC,#precisa adicionar pelo menos um empate para não dar erro de dimensionamento.
    met.cm(prev.mod.ufmg.comsimul.janela$resultado, tab$resul, FALSE)$MCC,
    met.cm(prev.mod.ufmg.comsimul.rodada$resultado, tab$resul, FALSE)$MCC,
    
    met.cm(prev.mod.ufmg.semsimul.ano$resultado, tab$resul, FALSE)$MCC,
    met.cm(prev.mod.ufmg.semsimul.geral$resultado, tab$resul, FALSE)$MCC,
    met.cm(prev.mod.ufmg.semsimul.janela$resultado, tab$resul, FALSE)$MCC,
    met.cm(prev.mod.ufmg.semsimul.rodada$resultado, tab$resul, FALSE)$MCC,
    
    met.cm(prev.mod.ufmg.simulnormal.ano$resultado, tab$resul, FALSE)$MCC,
    met.cm(prev.mod.ufmg.simulnormal.geral$resultado, tab$resul, FALSE)$MCC,
    met.cm(prev.mod.ufmg.simulnormal.janela$resultado, tab$resul, FALSE)$MCC,
    met.cm(prev.mod.ufmg.simulnormal.rodada$resultado, tab$resul, FALSE)$MCC
  )}
  
  #verossimilhança do resultado
  {verossimilhanca <- c(
    met.vero(tab$plac_mand, tab$plac_vist, prev.mod.controle.implicito.media$lambda1, prev.mod.controle.implicito.media$lambda2, F),
    "Não se aplica",
    "Não se aplica",
    "Não se aplica",
    "Não se aplica",
    "Não se aplica",
    "Não se aplica",
    
    met.vero(tab$plac_mand, tab$plac_vist, prev.mod.arr.chance1.ano$lambda1, prev.mod.arr.chance1.ano$lambda2, F),
    met.vero(tab$plac_mand, tab$plac_vist, prev.mod.arr.chance1.geral$lambda1, prev.mod.arr.chance1.geral$lambda2, F),
    met.vero(tab$plac_mand, tab$plac_vist, prev.mod.arr.chance1.janela$lambda1, prev.mod.arr.chance1.janela$lambda2, F),
    met.vero(tab$plac_mand, tab$plac_vist, prev.mod.arr.chance1.rodada$lambda1, prev.mod.arr.chance1.rodada$lambda2, F),
    
    met.vero(tab$plac_mand, tab$plac_vist, prev.mod.arr.chance2.ano$lambda1, prev.mod.arr.chance2.ano$lambda2, F),
    met.vero(tab$plac_mand, tab$plac_vist, prev.mod.arr.chance2.geral$lambda1, prev.mod.arr.chance2.geral$lambda2, F),
    met.vero(tab$plac_mand, tab$plac_vist, prev.mod.arr.chance2.janela$lambda1, prev.mod.arr.chance2.janela$lambda2, F),
    met.vero(tab$plac_mand, tab$plac_vist, prev.mod.arr.chance2.rodada$lambda1, prev.mod.arr.chance2.rodada$lambda2, F),
    
    met.vero(tab$plac_mand, tab$plac_vist, prev.mod.arr.SD0.ano$lambda1, prev.mod.arr.SD0.ano$lambda2, F),
    met.vero(tab$plac_mand, tab$plac_vist, prev.mod.arr.SD0.geral$lambda1, prev.mod.arr.SD0.geral$lambda2, F),
    met.vero(tab$plac_mand, tab$plac_vist, prev.mod.arr.SD0.janela$lambda1, prev.mod.arr.SD0.janela$lambda2, F),
    met.vero(tab$plac_mand, tab$plac_vist, prev.mod.arr.SD0.rodada$lambda1, prev.mod.arr.SD0.rodada$lambda2, F),
    
    met.vero(tab$plac_mand, tab$plac_vist, prev.mod.arr.SD1.ano$lambda1, prev.mod.arr.SD1.ano$lambda2, F),
    met.vero(tab$plac_mand, tab$plac_vist, prev.mod.arr.SD1.geral$lambda1, prev.mod.arr.SD1.geral$lambda2, F),
    met.vero(tab$plac_mand, tab$plac_vist, prev.mod.arr.SD1.janela$lambda1, prev.mod.arr.SD1.janela$lambda2, F),
    met.vero(tab$plac_mand, tab$plac_vist, prev.mod.arr.SD1.rodada$lambda1, prev.mod.arr.SD1.rodada$lambda2, F),
    
    "Não se aplica",
    "Não se aplica",
    "Não se aplica",
    "Não se aplica",
    
    "Não se aplica",
    "Não se aplica",
    "Não se aplica",
    "Não se aplica",
    
    "Não se aplica",
    "Não se aplica",
    "Não se aplica",
    "Não se aplica"
  )}
  
  #memoria dos modelos
  {memoria <- c(
    "Geral",     # mod.controle.implicito.media
    "Nula",  # mod.controle.mand
    "Nula",  # mod.controle.emp
    "Nula",  # mod.controle.vist
    "Geral",     # mod.controle.media
    "Nula",    # mod.controle.poisson
    "Nula",    # mod.controle.uniforme
    
    "Anual",     # mod.arr.chance1.ano
    "Geral",     # mod.arr.chance1.geral
    "Semestre",     # mod.arr.chance1.janela
    "Rodada",     # mod.arr.chance1.rodada
    
    "Anual",     # mod.arr.chance2.ano
    "Geral",     # mod.arr.chance2.geral
    "Semestre",     # mod.arr.chance2.janela
    "Rodada",     # mod.arr.chance2.rodada
    
    "Anual",     # mod.arr.SD0.ano
    "Geral",     # mod.arr.SD0.geral
    "Semestre",     # mod.arr.SD0.janela
    "Rodada",     # mod.arr.SD0.rodada
    
    "Anual",     # mod.arr.SD1.ano
    "Geral",     # mod.arr.SD1.geral
    "Semestre",     # mod.arr.SD1.janela
    "Rodada",     # mod.arr.SD1.rodada
    
    "Anual",    # mod.ufmg.comsimul.ano
    "Geral",    # mod.ufmg.comsimul.geral
    "Semestre",    # mod.ufmg.comsimul.janela
    "Rodada",    # mod.ufmg.comsimul.rodada
    
    "Anual",  # mod.ufmg.semsimul.ano
    "Geral",  # mod.ufmg.semsimul.geral
    "Semestre",  # mod.ufmg.semsimul.janela
    "Rodada",  # mod.ufmg.semsimul.rodada
    
    "Anual",  # mod.ufmg.simulnormal.ano
    "Geral",  # mod.ufmg.simulnormal.geral
    "Semestre",  # mod.ufmg.simulnormal.janela
    "Rodada"   # mod.ufmg.simulnormal.rodada
  )}
  
  #Autor
  {Autor <- c(
    "Arruda",     # mod.controle.implicito.media
    "Controle",  # mod.controle.mand
    "Controle",  # mod.controle.emp
    "Controle",  # mod.controle.vist
    "Controle",     # mod.controle.media
    "Controle",    # mod.controle.poisson
    "Controle",    # mod.controle.uniforme
    
    "Arruda",     # mod.arr.chance1.ano
    "Arruda",     # mod.arr.chance1.geral
    "Arruda",     # mod.arr.chance1.janela
    "Arruda",     # mod.arr.chance1.rodada
    
    "Arruda",     # mod.arr.chance2.ano
    "Arruda",     # mod.arr.chance2.geral
    "Arruda",     # mod.arr.chance2.janela
    "Arruda",     # mod.arr.chance2.rodada
    
    "Arruda",     # mod.arr.SD0.ano
    "Arruda",     # mod.arr.SD0.geral
    "Arruda",     # mod.arr.SD0.janela
    "Arruda",     # mod.arr.SD0.rodada
    
    "Arruda",     # mod.arr.SD1.ano
    "Arruda",     # mod.arr.SD1.geral
    "Arruda",     # mod.arr.SD1.janela
    "Arruda",     # mod.arr.SD1.rodada
    
    "Ufmg",    # mod.ufmg.comsimul.ano
    "Ufmg",    # mod.ufmg.comsimul.geral
    "Ufmg",    # mod.ufmg.comsimul.janela
    "Ufmg",    # mod.ufmg.comsimul.rodada
    
    "Ufmg",  # mod.ufmg.semsimul.ano
    "Ufmg",  # mod.ufmg.semsimul.geral
    "Ufmg",  # mod.ufmg.semsimul.janela
    "Ufmg",  # mod.ufmg.semsimul.rodada
    
    "Ufmg",  # mod.ufmg.simulnormal.ano
    "Ufmg",  # mod.ufmg.simulnormal.geral
    "Ufmg",  # mod.ufmg.simulnormal.janela
    "Ufmg"   # mod.ufmg.simulnormal.rodada
  )}
  
  # Converter a lista em um data.frame
  resultados_df <- data.frame(
    Mod = modelos,
    TA  = taxa_acerto,
    EPMP= as.numeric(epmp),
    MD  = as.numeric(definetti),
    MDD = as.numeric(definetti.detalhada),
    TF  = taxa_func,
    NC  = Complexidade,
    MC_Pre = as.numeric(mc_precision),
    MC_Rec = as.numeric(mc_recall),
    MC_F1  = as.numeric(mc_f1),
    MC_MCC = as.numeric(mc_MCC),
    Vero   = as.numeric(verossimilhanca),
    Memória = memoria,
    Autor   = Autor
  )
}
  # Exibir o resultado
  print(resultados_df)


    library(knitr)
  kable(resultados_df, format = "markdown", )
  col.names = c("Modelo", 
                "Taxa de Acerto", 
                "Erro preditivo ponderado médio", 
                "Medida de Definetti", 
                "Medida de Definetti detalhada",
                "Taxa de Funcionamento",
                "Nível de Complexidade",
                "MC - Precision",
                "MC - Recall",
                "MC - F1 Score",
                "MC - MCC",
                "Verossimilhança",
                "Memória",
                "Autor"
                )
  
  sd(na.omit((resultados_df$TA)))
  sd(na.omit((resultados_df$EPMP)))
  sd(na.omit((resultados_df$MD)))
  sd(na.omit((resultados_df$MDD)))
  sd(na.omit((resultados_df$TF)))
  sd(na.omit((resultados_df$NC)))
  sd(na.omit((resultados_df$MC_Pre)))
  sd(na.omit((resultados_df$MC_Rec)))
  sd(na.omit((resultados_df$MC_F1)))
  sd(na.omit((resultados_df$MC_MCC)))
  sd(na.omit((resultados_df$Vero)))
  
  summary(resultados_df)
  
  #cor dos modelos
  {Cor <- c(
    "olivedrab",     # mod.controle.implicito.media
    "olivedrab",  # mod.controle.mand
    "olivedrab",  # mod.controle.emp
    "olivedrab",  # mod.controle.vist
    "olivedrab",     # mod.controle.media
    "olivedrab",    # mod.controle.poisson
    "olivedrab",    # mod.controle.uniforme
    
    "slateblue",     # mod.arr.chance1.ano
    "slateblue",     # mod.arr.chance1.geral
    "slateblue",     # mod.arr.chance1.janela
    "slateblue",     # mod.arr.chance1.rodada
    
    "dodgerblue3",     # mod.arr.chance2.ano
    "dodgerblue3",     # mod.arr.chance2.geral
    "dodgerblue3",     # mod.arr.chance2.janela
    "dodgerblue3",     # mod.arr.chance2.rodada
    
    "aquamarine",     # mod.arr.SD0.ano
    "aquamarine",     # mod.arr.SD0.geral
    "aquamarine",     # mod.arr.SD0.janela
    "aquamarine",     # mod.arr.SD0.rodada
    
    "turquoise4",     # mod.arr.SD1.ano
    "turquoise4",     # mod.arr.SD1.geral
    "turquoise4",     # mod.arr.SD1.janela
    "turquoise4",     # mod.arr.SD1.rodada
    
    "brown",    # mod.ufmg.comsimul.ano
    "brown",    # mod.ufmg.comsimul.geral
    "brown",    # mod.ufmg.comsimul.janela
    "brown",    # mod.ufmg.comsimul.rodada
    
    "goldenrod2",  # mod.ufmg.semsimul.ano
    "goldenrod2",  # mod.ufmg.semsimul.geral
    "goldenrod2",  # mod.ufmg.semsimul.janela
    "goldenrod2",  # mod.ufmg.semsimul.rodada
    
    "chocolate3",  # mod.ufmg.simulnormal.ano
    "chocolate3",  # mod.ufmg.simulnormal.geral
    "chocolate3",  # mod.ufmg.simulnormal.janela
    "chocolate3"   # mod.ufmg.simulnormal.rodada
  )}
  
  resultados_df <- resultados_df |>
    mutate(Cor = Cor) |>
    mutate(Mod = factor(Mod, levels = Mod))  # reordenação dos modelos
  
  # Criar uma coluna com o nome da família
  resultados_df$Familia <- with(resultados_df, case_when(
    Cor == "olivedrab"   ~ "Controle",
    Cor == "slateblue"   ~ "Chance I",
    Cor == "dodgerblue3" ~ "Chance II",
    Cor == "aquamarine"  ~ "SD 0",
    Cor == "turquoise4"  ~ "SD 1",
    Cor == "brown"       ~ "UFMG original",
    Cor == "goldenrod2"  ~ "UFMG sem sor.",
    Cor == "chocolate3"  ~ "UFMG sor. normal",
    TRUE ~ "Outro"
  ))
  
  # Criar uma coluna com o nome da família
  resultados_df$textura <- with(resultados_df, case_when(
    Memória == "Geral"    ~ "crosshatch",
    Memória == "Anual"    ~ "stripe",
    Memória == "Semestre" ~ "circle",
    Memória == "Rodada"   ~ "none",
    Memória == "Nula"     ~ "wave"
  ))
  
  transparente <- rgb(1, 0, 0, alpha = 0)
  #write.csv(tab_metricas, "tabela_resultados_das_metricas2.csv", row.names = FALSE)
}

tab_teste <- select(resultados_df, Mod, TA, MD, MC_F1, Memória, Familia)

###graficos medidas absolutas ---------------------
# plotar grafico da TA
{
  ggplot(resultados_df, aes(x = TA, y = reorder(Mod, TA), fill = Familia)) +
    geom_bar(stat = "identity") +
    geom_text(aes(label = round(TA, 3)), hjust = -0.1, size = 3.8) +
    labs(
      x = "Taxa de Acerto",
      y = "Modelo",
      fill = "Família de Modelos"
    ) +
    scale_fill_manual(values = setNames(resultados_df$Cor, resultados_df$Familia)) +
    scale_x_continuous(
      breaks = seq(0, 0.5, by = 0.05),
      expand = expansion(mult = c(0, 0.1))
    ) +
    theme_minimal() +
    theme(
      panel.grid.major = element_line(color = "gray65", linewidth = 0.25),
      panel.grid.minor = element_blank(),
      axis.title = element_text(size = 12),
      axis.text = element_text(size = 10),
      axis.text.y = element_text(
        angle = 0,
        hjust = 0,
        vjust = 0.5,
        lineheight = 2
      )
    )

# Exportar o gráfico (ajuste tamanho conforme quantidade de modelos)
ggsave("graf_taxa_acerto.png", plot = last_plot(),
       width = 19, height = 15, units = "cm", dpi = 300)
}

##### gráfico EPMP
{# Ajuste da base
df_epmp <- resultados_df |>
  select(Modelo = Mod, EPMP) |>
  mutate(
    Cor = ifelse(is.na(EPMP), transparente, Cor),
    Familia = ifelse(is.na(EPMP), "", resultados_df$Familia),
    label = ifelse(is.na(EPMP), "Não se aplica", round(EPMP, 3)),
    ajuste = ifelse(is.na(EPMP), 4.35, -0.1),
    EPMP_plot = ifelse(is.na(EPMP), max(EPMP, na.rm = TRUE) + 0.01, EPMP)
  ) |>
  arrange(EPMP_plot) |>
  mutate(Modelo = factor(Modelo, levels = Modelo))  # reordenação final

ggplot(df_epmp, aes(x = EPMP_plot, y = reorder(Modelo, -EPMP_plot), fill = Familia)) +
  geom_bar(stat = "identity") +
  geom_text(aes(label = label, hjust = ajuste), size = 3.8) +
  scale_fill_manual(values = setNames(df_epmp$Cor, df_epmp$Familia)) +
  labs(
    x = "Erro Preditivo Médio Ponderado",
    y = "Modelo",
    fill = "Família de Modelos"
  ) +
  scale_x_continuous(
    breaks = seq(0, 1, by = 0.1),
    expand = expansion(mult = c(0, 0.1))
  ) +
  theme_minimal() +
  theme(
    panel.grid.major = element_line(color = "gray65", linewidth = 0.25),
    panel.grid.minor = element_blank(),
    axis.title = element_text(size = 12),
    axis.text = element_text(size = 11),
    axis.text.y = element_text(
      angle = 0,
      hjust = 0,
      vjust = 0.5,
      lineheight = 2
    )
  )

# Exportar o gráfico (ajuste tamanho conforme quantidade de modelos)
ggsave("graf_epmp.png", plot = last_plot(),
       width = 19, height = 15, units = "cm", dpi = 300)
}

### gráfico MD
{# Ajuste da base
  df_md <- resultados_df |>
    select(Modelo = Mod, MD, Cor, Familia) |>
    mutate(
      Cor = ifelse(is.na(MD), transparente, Cor),  # branco para NA, cor fixa para valores
      Familia = ifelse(is.na(MD), "", Familia),      # sem nome de família para NA
      label = ifelse(is.na(MD), "Não se aplica", round(MD, 3)),
      ajuste = ifelse(is.na(MD), 4.35, -0.1),
      MD_plot = ifelse(is.na(MD), max(MD, na.rm = TRUE) + 0.01, MD)
    ) |>
    arrange(MD_plot) |>
    mutate(Modelo = factor(Modelo, levels = Modelo))  # reordenação final
  
  # Gráfico
  ggplot(df_md, aes(x = MD_plot, y = reorder(Modelo, -MD_plot), fill = Familia)) +
    geom_bar(stat = "identity") +
    geom_text(aes(label = label, hjust = ajuste), size = 3.8) +
    scale_fill_manual(values = setNames(df_md$Cor, df_md$Familia)) +
    labs(
      x = "Medida de De Finetti",
      y = "Modelo",
      fill = "Família de Modelos"
    ) +
    scale_x_continuous(
      breaks = seq(0, 1.6, by = 0.2),
      expand = expansion(mult = c(0, 0.1))
    ) +
    theme_minimal() +
    theme(
      panel.grid.major = element_line(color = "gray65", linewidth = 0.25),
      panel.grid.minor = element_blank(),
      axis.title = element_text(size = 12),
      axis.text = element_text(size = 11),
      axis.text.y = element_text(
        angle = 0,
        hjust = 0,
        vjust = 0.5,
        lineheight = 2
      )
    )

  ggsave("graf_md.png", plot = last_plot(),
         width = 19, height = 15, units = "cm", dpi = 300)
}

#### gráfico MDD
{####
# Ajuste da base
df_mdd <- resultados_df |>
  select(Modelo = Mod, MDD, Cor, Familia) |>
  mutate(
    Cor = ifelse(is.na(MDD), transparente, Cor),  # cor transparente se MDD for NA
    Familia = ifelse(is.na(MDD), "", Familia), # sem nome de família para NA
    label = ifelse(is.na(MDD), "Não se aplica", round(MDD, 3)),  # texto adequado
    ajuste = ifelse(is.na(MDD), 4.85, -0.1),  # posição horizontal do texto
    MDD_plot = ifelse(is.na(MDD), max(MDD, na.rm = TRUE) + 0.01, MDD)  # mover NAs para o final
  ) |>
  arrange(MDD_plot) |>
  mutate(Modelo = factor(Modelo, levels = Modelo)) # reordenação final

# Gráfico
ggplot(df_mdd, aes(x = MDD_plot, y = reorder(Modelo, -MDD_plot), fill = Familia)) +
  geom_bar(stat = "identity") +
  geom_text(aes(label = label, hjust = ajuste), size = 3.8) +
  scale_fill_manual(values = setNames(df_md$Cor, df_md$Familia)) +
  labs(
    x = "Medida de Definetti Detalhada",
    y = "Modelo",
    title = NULL
  ) +
  scale_x_continuous(
    breaks = seq(0, 3, by = 0.4),
    expand = expansion(mult = c(0, 0.1))
  ) +
  theme_minimal() +
  theme(
    panel.grid.major = element_line(color = "gray65", linewidth = 0.25),
    panel.grid.minor = element_blank(),
    axis.title = element_text(size = 12),
    axis.text = element_text(size = 11),
    axis.text.y = element_text(
      angle = 0,
      hjust = 0,
      vjust = 0.5,
      lineheight = 2
    )
  )

# Exportar o gráfico (ajuste tamanho conforme quantidade de modelos)
ggsave("graf_mdd.png", plot = last_plot(), width = 19, height = 15, units = "cm", dpi = 300)
}

#### gráfico NC
{####
  # Ajuste da base
  df_nc <- resultados_df |>
    select(Modelo = Mod, NC, Cor, Familia) |>
    mutate(
      Cor = ifelse(is.na(NC), transparente, Cor),  # cor transparente se MDD for NA
      Familia = ifelse(is.na(NC), "", Familia), # sem nome de família para NA
      label = ifelse(is.na(NC), "Não se aplica", round(NC, 3)),  # texto adequado
      ajuste = ifelse(is.na(NC), 4.85, -0.1),  # posição horizontal do texto
      NC_plot = ifelse(is.na(NC), max(NC, na.rm = TRUE) + 0.01, NC)  # mover NAs para o final
    ) |>
    arrange(NC_plot) |>
    mutate(Modelo = factor(Modelo, levels = Modelo)) # reordenação final
  
  # Gráfico
  ggplot(df_nc, aes(x = NC_plot, y = reorder(Modelo, -NC_plot), fill = Familia)) +
    geom_bar(stat = "identity") +
    geom_text(aes(label = label, hjust = ajuste), size = 3.8) +
    scale_fill_manual(values = setNames(df_nc$Cor, df_nc$Familia)) +
    labs(
      x = "Nível de Complexidade",
      y = "Modelo",
      title = NULL
    ) +
    scale_x_continuous(
      breaks = seq(0, 6, by = 1),
      expand = expansion(mult = c(0, 0.1))
    ) +
    theme_minimal() +
    theme(
      panel.grid.major = element_line(color = "gray65", linewidth = 0.25),
      panel.grid.minor = element_blank(),
      axis.title = element_text(size = 12),
      axis.text = element_text(size = 11),
      axis.text.y = element_text(
        angle = 0,
        hjust = 0,
        vjust = 0.5,
        lineheight = 2
      )
    )
  
  # Exportar o gráfico (ajuste tamanho conforme quantidade de modelos)
  ggsave("graf_nc.png", plot = last_plot(), width = 19, height = 15, units = "cm", dpi = 300)
}

### gráfico Vero
{df_vero <- resultados_df |>
    select(Modelo = Mod, Vero, Cor, Familia) |>
    mutate(
      Cor = ifelse(is.na(Vero), transparente, Cor),  # cor transparente em NA
      Familia = ifelse(is.na(Vero), "", Familia),    # sem família para NA
      label = ifelse(is.na(Vero), "Não se aplica", round(Vero, 4)),
      ajuste = ifelse(is.na(Vero), 1.77, -0.1),
      
      # Para NA, posiciono levemente fora do eixo
      Vero_plot = ifelse(is.na(Vero), min(Vero, na.rm = TRUE) - 0.01, Vero)
    ) |>
    arrange(Vero_plot) |>
    mutate(Modelo = factor(Modelo, levels = Modelo))
  
  
  # Gráfico
  ggplot(df_vero, aes(x = Vero_plot, y = reorder(Modelo, Vero_plot), fill = Familia)) +
    geom_bar(stat = "identity") +
    geom_text(aes(label = label, hjust = ajuste), size = 3.8) +
    scale_fill_manual(values = setNames(df_vero$Cor, df_vero$Familia)) +
    labs(
      x = "Verossimilhança",
      y = "Modelo",
      fill = "Família de Modelos"
    ) +
    scale_x_continuous(
      breaks = seq(0, 0.08, by = 0.02),
      expand = expansion(mult = c(0, 0.1))
    ) +
    theme_minimal() +
    theme(
      panel.grid.major = element_line(color = "gray65", linewidth = 0.25),
      panel.grid.minor = element_blank(),
      axis.title = element_text(size = 12),
      axis.text = element_text(size = 11),
      axis.text.y = element_text(
        angle = 0,
        hjust = 0,
        vjust = 0.5,
        lineheight = 2
      )
    )
  
  ggsave("graf_vero.png", plot = last_plot(), width = 19, height = 15, units = "cm", dpi = 300)
}

##### gráfico MC_Pre (Precisão)
{df_mcpre <- resultados_df |>
  select(Modelo = Mod, MC_Pre, Cor, Familia) |>
  mutate(
    Cor = ifelse(is.na(MC_Pre), transparente, Cor),
    Familia = ifelse(is.na(MC_Pre), "", Familia),
    label = ifelse(is.na(MC_Pre), "Não se aplica", round(MC_Pre, 3)),
    ajuste = ifelse(is.na(MC_Pre), 2.48, -0.1),
    MC_Pre_plot = ifelse(is.na(MC_Pre), min(MC_Pre, na.rm = TRUE) - 0.01, MC_Pre)
  ) |>
  arrange(MC_Pre_plot) |>
  mutate(Modelo = factor(Modelo, levels = Modelo))  # reordenação final

# Gráfico
ggplot(df_mcpre, aes(x = MC_Pre_plot, y = reorder(Modelo, MC_Pre_plot), fill = Familia)) +
  geom_bar(stat = "identity") +
  geom_text(aes(label = label, hjust = ajuste), size = 3.8) +
  scale_fill_manual(values = setNames(df_mcpre$Cor, df_mcpre$Familia)) +
  labs(
    x = "Precisão",
    y = "Modelo",
    fill = "Família de Modelos"
  ) +
  scale_x_continuous(
    breaks = seq(0, 1, by = 0.1),
    expand = expansion(mult = c(0, 0.1))
  ) +
  theme_minimal() +
  theme(
    panel.grid.major = element_line(color = "gray65", linewidth = 0.25),
    panel.grid.minor = element_blank(),
    axis.title = element_text(size = 12),
    axis.text = element_text(size = 11),
    axis.text.y = element_text(
      angle = 0,
      hjust = 0,
      vjust = 0.5,
      lineheight = 2
    )
  )

# Exportar o gráfico (ajuste tamanho conforme quantidade de modelos)
ggsave("graf_mcpre.png", plot = last_plot(), width = 19, height = 15, units = "cm", dpi = 300)
}

##### gráfico MC_Rec (Recall)
{df_mcrec <- resultados_df |>
  select(Modelo = Mod, MC_Rec, Cor, Familia) |>
  mutate(
    Cor = ifelse(is.na(MC_Rec), transparente, Cor),
    Familia = ifelse(is.na(MC_Rec), "", Familia),
    label = ifelse(is.na(MC_Rec), "Não se aplica", round(MC_Rec, 3)),
    ajuste = ifelse(is.na(MC_Rec), 0, -0.1),
    MC_Rec_plot = ifelse(is.na(MC_Rec), 0.000001, MC_Rec)
  ) |>
  arrange(MC_Rec_plot) |>
  mutate(Modelo = factor(Modelo, levels = Modelo))

ggplot(df_mcrec, aes(x = MC_Rec_plot, y = reorder(Modelo, MC_Rec_plot), fill = Familia)) +
  geom_bar(stat = "identity") +
  geom_text(aes(label = label, hjust = ajuste), size = 3.8) +
  scale_fill_manual(values = setNames(df_mcrec$Cor, df_mcrec$Familia)) +
  labs(
    x = "Recall",
    y = "Modelo",
    fill = "Família de Modelos"
  ) +
  scale_x_continuous(
    breaks = seq(0, 1, by = 0.1),
    expand = expansion(mult = c(0, 0.1))
  ) +
  theme_minimal() +
  theme(
    panel.grid.major = element_line(color = "gray65", linewidth = 0.25),
    panel.grid.minor = element_blank(),
    axis.title = element_text(size = 12),
    axis.text = element_text(size = 11),
    axis.text.y = element_text(
      angle = 0,
      hjust = 0,
      vjust = 0.5,
      lineheight = 2
    )
  )

ggsave("graf_mcrec.png", plot = last_plot(), width = 19, height = 15, units = "cm", dpi = 300)
}

##### gráfico MC_F1 (F1-Score)
{df_mcf1 <- resultados_df |>
  select(Modelo = Mod, MC_F1, Cor, Familia) |>
  mutate(
    Cor = ifelse(is.na(MC_F1), transparente, Cor),
    Familia = ifelse(is.na(MC_F1), "", Familia),
    label = ifelse(is.na(MC_F1), "Não se aplica", round(MC_F1, 3)),
    ajuste = ifelse(is.na(MC_F1), 0, -0.1),
    MC_F1_plot = ifelse(is.na(MC_F1), 0.0001, MC_F1)
  ) |>
  arrange(MC_F1_plot) |>
  mutate(Modelo = factor(Modelo, levels = Modelo))
  # Gráfico
ggplot(df_mcf1, aes(x = MC_F1_plot, y = reorder(Modelo, MC_F1_plot), fill = Familia)) +
  geom_bar(stat = "identity") +
  geom_text(aes(label = label, hjust = ajuste), size = 3.8) +
  scale_fill_manual(values = setNames(df_mcf1$Cor, df_mcf1$Familia)) +
  labs(
    x = "F1-Score",
    y = "Modelo",
    fill = "Família de Modelos"
  ) +
  scale_x_continuous(
    breaks = seq(0, 1, by = 0.1),
    expand = expansion(mult = c(0, 0.1))
  ) +
  theme_minimal() +
  theme(
    panel.grid.major = element_line(color = "gray65", linewidth = 0.25),
    panel.grid.minor = element_blank(),
    axis.title = element_text(size = 12),
    axis.text = element_text(size = 11),
    axis.text.y = element_text(
      angle = 0,
      hjust = 0,
      vjust = 0.5,
      lineheight = 2
    )
  )

ggsave("graf_mcf1.png", plot = last_plot(), width = 19, height = 15, units = "cm", dpi = 300)
}

##### gráfico MC_MCC (Matthews Correlation Coefficient)
{df_mcmcc <- resultados_df |>
  select(Modelo = Mod, MC_MCC, Cor, Familia) |>
  mutate(
    Cor = ifelse(is.na(MC_MCC), transparente, Cor),
    Familia = ifelse(is.na(MC_MCC), "", Familia),
    label = ifelse(is.na(MC_MCC), "Não se aplica", round(MC_MCC, 3)),
    ajuste = ifelse(is.na(MC_MCC), -1.05, ifelse(MC_MCC<0, 1, -0.15)),
    MC_MCC_plot = ifelse(is.na(MC_MCC), min(MC_MCC, na.rm = TRUE) - 0.006, MC_MCC)
  ) |>
  arrange(MC_MCC_plot) |>
  mutate(Modelo = factor(Modelo, levels = Modelo))

ggplot(df_mcmcc, aes(x = MC_MCC_plot, y = reorder(Modelo, MC_MCC_plot), fill = Familia)) +
  geom_bar(stat = "identity") +
  geom_text(aes(label = label, hjust = ajuste), size = 3.8) +
  scale_fill_manual(values = setNames(df_mcmcc$Cor, df_mcmcc$Familia)) +
  labs(
    x = "Coeficiente de Matthews",
    y = "Modelo",
    fill = "Família de Modelos"
  ) +
  scale_x_continuous(
    breaks = seq(-0.03, 0.07, by = 0.01),
    expand = expansion(mult = c(0.01, 0.1))
  ) +
  theme_minimal() +
  theme(
    panel.grid.major = element_line(color = "gray65", linewidth = 0.25),
    panel.grid.minor = element_blank(),
    axis.title = element_text(size = 12),
    axis.text = element_text(size = 11),
    axis.text.y = element_text(
      angle = 0,
      hjust = 0,
      vjust = 0.5,
      lineheight = 2
    )
  )

ggsave("graf_mcmcc.png", plot = last_plot(), width = 20, height = 15, units = "cm", dpi = 300)
}

##  graficos  de duas variaveis -------------------------------

#### gráfico TA e NC
{# Gráfico com nomes dos modelos
  ggplot(resultados_df, aes(x = NC, y = TA)) +
    geom_text(aes(label = Mod), size = 4.2, color = "deepskyblue4") +
    labs(
      x = "Nível de Complexidade",
      y = "Taxa de Acerto",
      title = NULL
    ) +
    scale_x_continuous(expand = expansion(mult = c(0.05, 0.05))) +
    scale_y_continuous(
      breaks = seq(0, 1, by = 0.05),  # ← Aqui define divisões de 0.05
      expand = expansion(mult = c(0.05, 0.05))
    ) +
    theme_minimal() +
    theme(
      axis.title = element_text(size = 12),
      axis.text = element_text(size = 11),
      panel.grid.major = element_line(color = "gray80"),
      panel.grid.minor = element_line(color = "gray80")
    )
  
  ggsave("graf_ncxta.png", plot = last_plot(), width = 25, height = 15, units = "cm", dpi = 300)
}

## Graficos com variaveis agrupadas Memoria--------------------

#### graf. TA agrupado
{  metrica <- "TA"   # <-- troque aqui para MC_MCC, TA, DF, etc.
    df_plot <- resultados_df |>
    select(
      Modelo = Mod,
      Valor = !!sym(metrica),
      Memoria = Memória,
      Cor,
      Familia
    ) |>
    mutate(
      Memoria = factor(
        Memoria,
        levels = c("Geral","Anual","Semestre","Rodada","Nula")
      ),
      # Ajustes para lidar com NA (como no MD)
      Cor = ifelse(is.na(Valor), transparente, Cor),
      Familia = ifelse(is.na(Valor), "", Familia),
      label = ifelse(is.na(Valor), "Não se aplica", round(Valor, 3)),
      ajuste = ifelse(is.na(Valor), 0, 0.1),
      Valor_plot = ifelse(is.na(Valor), max(Valor, na.rm = TRUE) + 0.01, Valor)
    ) |>
    arrange(Valor_plot) |>
    mutate(Modelo = factor(Modelo, levels = Modelo))
  
  #---------------------------------------------
  # Paleta nomeada
  #---------------------------------------------
  cores_fam <- df_plot |>
    distinct(Familia, Cor) |>
    (\(x) setNames(x$Cor, x$Familia))()
  
  #---------------------------------------------
  # GRÁFICO FINAL
  #---------------------------------------------
  g_metric <- ggplot(df_plot, aes(x = Valor_plot, y = reorder(Modelo, Valor_plot), fill = Familia)) +
    geom_col() +
    geom_text(aes(label = label, x = Valor_plot + 0.01),
              size = 3.5, hjust = 0) +
    labs(
      #title = paste0("Desempenho da Métrica: ", metrica),
      x = metrica,
      y = "Modelo",
      fill = "Família de Modelos"
    ) +
    scale_fill_manual(values = cores_fam) +
    scale_x_continuous(
      expand = expansion(mult = c(0, 0.1))
    ) +
    facet_wrap(~ Memoria, ncol = 1, scales = "free_y") +
    theme_minimal(base_size = 13) +
    theme(
      panel.grid.major = element_line(color = "grey70", linewidth = 0.6),
      panel.grid.minor = element_line(color = "grey80", linewidth = 0.4),
      legend.position = "right",
      strip.text = element_text(size = 14, face = "bold"),
      axis.text.y = element_text(size = 10),
      axis.text.x = element_text(size = 10)
    )
  
  g_metric
  # Exportar o gráfico (ajuste tamanho conforme quantidade de modelos)
  ggsave("graf_TA_mem.png", plot = g_metric, width = 16, height = 20, units = "cm", dpi = 300)
}

#### graf. MD agrupado
{metrica <- "MD"   # <-- troque aqui para MC_MCC, TA, DF, etc.
  df_plot <- resultados_df |>
    select(
      Modelo = Mod,
      Valor = !!sym(metrica),
      Memoria = Memória,
      Cor,
      Familia
    ) |>
    mutate(
      Memoria = factor(
        Memoria,
        levels = c("Geral","Anual","Semestre","Rodada","Nula")
      ),
      # Ajustes para lidar com NA (como no MD)
      Cor = ifelse(is.na(Valor), transparente, Cor),
      Familia = ifelse(is.na(Valor), "", Familia),
      label = ifelse(is.na(Valor), "Não se aplica", round(Valor, 3)),
      ajuste = ifelse(is.na(Valor), 3.15, 0),
      Valor_plot = ifelse(is.na(Valor), 1.6, Valor)
    ) |>
    arrange(Valor_plot) |>
    mutate(Modelo = factor(Modelo, levels = Modelo))

  cores_fam <- df_plot |>
    distinct(Familia, Cor) |>
    (\(x) setNames(x$Cor, x$Familia))()
  #---------------------------------------------
  # GRÁFICO FINAL
  #---------------------------------------------
  g_metric <- ggplot(df_plot, aes(x = Valor_plot, y = reorder(Modelo, -Valor_plot), fill = Familia)) +
    geom_col() +
    geom_text(aes(label = label, x = Valor_plot + 0.01),
              size = 3.5, hjust = df_plot$ajuste) +
    labs(
      #title = paste0("Desempenho da Métrica: ", metrica),
      x = metrica,
      y = "Modelo",
      fill = "Família de Modelos"
    ) +
    scale_fill_manual(values = cores_fam) +
    scale_x_continuous(
      expand = expansion(mult = c(0, 0.1))
    ) +
    facet_wrap(~ Memoria, ncol = 1, scales = "free_y") +
    theme_minimal(base_size = 13) +
    theme(
      panel.grid.major = element_line(color = "grey70", linewidth = 0.6),
      panel.grid.minor = element_line(color = "grey80", linewidth = 0.4),
      legend.position = "right",
      strip.text = element_text(size = 14, face = "bold"),
      axis.text.y = element_text(size = 10),
      axis.text.x = element_text(size = 10)
    )
  g_metric
  # Exportar o gráfico (ajuste tamanho conforme quantidade de modelos)
  ggsave("graf_MD_mem.png", plot = g_metric, width = 16, height = 20, units = "cm", dpi = 300)
}

#### graf. F1 agrupado
{metrica <- "MC_F1"   # <-- troque aqui para MC_MCC, TA, DF, etc.
  
  df_plot <- resultados_df |>
    select(
      Modelo = Mod,
      Valor = !!sym(metrica),
      Memoria = Memória,
      Cor,
      Familia
    ) |>
    mutate(
      Memoria = factor(
        Memoria,
        levels = c("Geral","Anual","Semestre","Rodada","Nula")
      ),
      # Ajustes para lidar com NA (como no MD)
      Cor = ifelse(is.na(Valor), transparente, Cor),
      Familia = ifelse(is.na(Valor), "", Familia),
      label = ifelse(is.na(Valor), "Não se aplica", round(Valor, 3)),
      ajuste = ifelse(is.na(Valor), 0.08, 0.15),
      Valor_plot = ifelse(is.na(Valor), 0, Valor)
    ) |>
    arrange(Valor_plot) |>
    mutate(Modelo = factor(Modelo, levels = Modelo))
  #---------------------------------------------
  # Paleta nomeada
  #---------------------------------------------
  cores_fam <- df_plot |>
    distinct(Familia, Cor) |>
    (\(x) setNames(x$Cor, x$Familia))()
  #---------------------------------------------
  # GRÁFICO FINAL
  #---------------------------------------------
  g_metric <- ggplot(df_plot, aes(x = Valor_plot, y = reorder(Modelo, Valor_plot), fill = Familia)) +
    geom_col() +
    geom_text(aes(label = label, x = Valor_plot + 0.01),
              size = 3.5, hjust = df_plot$ajuste) +
    labs(
      #title = paste0("Desempenho da Métrica: ", metrica),
      x = metrica,
      y = "Modelo",
      fill = "Família de Modelos"
    ) +
    scale_fill_manual(values = cores_fam) +
    scale_x_continuous(
      expand = expansion(mult = c(0, 0.1))
    ) +
    facet_wrap(~ Memoria, ncol = 1, scales = "free_y") +
    theme_minimal(base_size = 13) +
    theme(
      panel.grid.major = element_line(color = "grey70", linewidth = 0.6),
      panel.grid.minor = element_line(color = "grey80", linewidth = 0.4),
      legend.position = "right",
      strip.text = element_text(size = 14, face = "bold"),
      axis.text.y = element_text(size = 10),
      axis.text.x = element_text(size = 10)
    )
  g_metric
  # Exportar o gráfico (ajuste tamanho conforme quantidade de modelos)
  ggsave("graf_F1_mem.png", plot = g_metric, width = 16.5, height = 20, units = "cm", dpi = 300)
}


## Graficos com variaveis agrupadas Familia --------------------

#### graf. TA agrupado
{  metrica <- "TA" 
  #titulo <- paste0("Desempenho da Métrica ", metrica, " Agrupado por Família") 
  titulo <- NULL
  
  df <- resultados_df |> 
    select( Modelo = Mod, Metrica = all_of(metrica), Cor, Familia ) |>
    mutate( Familia = factor(Familia, 
                             levels = c("SD 0", "SD 1", "Chance I", "Chance II", "UFMG original", "UFMG sem sor.", "UFMG sor. normal", "Controle" )),
            # Modelos sem valor 
            Cor = ifelse(is.na(Metrica), transparente, Cor), 
            label = ifelse(is.na(Metrica), "Não se aplica", round(Metrica, 3)), 
            ajuste = ifelse(is.na(Metrica), 4.35, -0.1),
            Metrica_plot = ifelse(is.na(Metrica), 
                                  max(Metrica, na.rm = TRUE) + 0.01, 
                                  Metrica) ) 
  #Paleta nomeada por família 
  cores_fam <- df |> 
    distinct(Familia, Cor) |> 
    (\(x) setNames(x$Cor, x$Familia))() 
  #----------------------------- # GRÁFICO FINAL #----------------------------- 
  g <- ggplot(df, aes(x = Metrica_plot, 
                      y = reorder(Modelo, Metrica_plot), 
                      fill = Familia)) + geom_col() + 
    geom_text(aes(label = label, 
                  x = Metrica_plot + 0.01), 
              hjust = 0, 
              size = 2.9) + 
    scale_fill_manual(values = cores_fam) + 
    labs( title = titulo, x = metrica, y = "Modelo" ) + 
    scale_x_continuous( breaks = scales::pretty_breaks(6), 
                        expand = expansion(mult = c(0, 0.1)) ) +
    facet_wrap(~ Familia, ncol = 1, scales = "free_y") + 
    theme_minimal(base_size = 13) + 
    theme( panel.grid.major = element_line(color = "grey70", linewidth = 0.6), 
           panel.grid.minor = element_line(color = "grey80", linewidth = 0.4), 
           legend.position = "none", strip.text = element_text(size = 14, face = "bold"), 
           axis.text.y = element_text(size = 8), axis.text.x = element_text(size = 10) ) 
  g
  # Exportar o gráfico (ajuste tamanho conforme quantidade de modelos)
  ggsave("graf_TA_fam.png", plot = g, width = 10, height = 20, units = "cm", dpi = 300)
}

#### graf. MD agrupado
{ metrica <- "MD" 
  #titulo <- paste0("Desempenho da Métrica ", metrica, " Agrupado por Família") 
  titulo <- NULL
  df <- resultados_df |> 
    select( Modelo = Mod, Metrica = all_of(metrica), Cor, Familia ) |>
    mutate( Familia = factor(Familia, 
                             levels = c("SD 0", "SD 1", "Chance I", "Chance II", "UFMG original", "UFMG sem sor.", "UFMG sor. normal", "Controle" )),
            Cor = ifelse(is.na(Metrica), transparente, Cor), 
            label = ifelse(is.na(Metrica), "Não se aplica", round(Metrica, 3)), 
            ajuste = ifelse(is.na(Metrica), 3.81, 0),
            Metrica_plot = ifelse(is.na(Metrica), 
                                  max(Metrica, na.rm = TRUE) + 0.01, 
                                  Metrica) ) 
  # Paleta nomeada pelas cores únicas
  cores_fam <- df |> 
    distinct(Cor, Familia) |> 
    (\(x) setNames(x$Cor, x$Cor))()   # nomeia por Cor, não por Família
  
  g <- ggplot(df, aes(
    x = Metrica_plot,
    y = reorder(Modelo, -Metrica_plot),
    fill = Cor                       # <<< CORREÇÃO AQUI
  )) +
    geom_col() +
    geom_text(
      aes(label = label, x = Metrica_plot + 0.01),
      hjust = df$ajuste, size = 2.8
    ) +
    scale_fill_manual(values = cores_fam) +   # <<< agora funciona
    labs(title = titulo, x = metrica, y = "Modelo") +
    scale_x_continuous(
      breaks = seq(0, 1.7, by = 0.2),
      expand = expansion(mult = c(0, 0.1))
    ) +
    facet_wrap(~ Familia, ncol = 1, scales = "free_y") +
    theme_minimal(base_size = 13) +
    theme(
      panel.grid.major = element_line(color = "grey70", linewidth = 0.6),
      panel.grid.minor = element_line(color = "grey80", linewidth = 0.4),
      legend.position = "none",
      strip.text = element_text(size = 13, face = "bold"),
      axis.text.y = element_text(size = 7),
      axis.text.x = element_text(size = 10)
    )
  g
  # Exportar o gráfico (ajuste tamanho conforme quantidade de modelos)
  ggsave("graf_MD_fam.png", plot = g, width = 10, height = 20, units = "cm", dpi = 300)
}

#### graf. F1 agrupado
{metrica <- "MC_F1"
  titulo <- NULL
  df <- resultados_df |> 
    select(Modelo = Mod, Metrica = all_of(metrica), Cor, Familia) |>
    mutate(
      Familia = factor(
        Familia,
        levels = c("SD 0", "SD 1", "Chance I", "Chance II",
                   "UFMG original", "UFMG sem sor.", "UFMG sor. normal", "Controle")
      ),
      Cor = ifelse(is.na(Metrica), transparente, Cor),
      label = ifelse(is.na(Metrica), "Não se aplica", round(Metrica, 3)),
      ajuste = ifelse(is.na(Metrica), 0.1, 0),
      Metrica_plot = ifelse(
        is.na(Metrica),
        0,
        Metrica
      )
    )
  
  # Paleta nomeada pelas cores únicas
  cores_fam <- df |> 
    distinct(Cor, Familia) |> 
    (\(x) setNames(x$Cor, x$Cor))()
  
  g <- ggplot(df, aes(
    x = Metrica_plot,
    y = reorder(Modelo, Metrica_plot),
    fill = Cor
  )) +
    geom_col() +
    geom_text(
      aes(label = label, x = Metrica_plot + 0.01),
      hjust = df$ajuste, size = 2.8
    ) +
    scale_fill_manual(values = cores_fam) +
    labs(title = titulo, x = metrica, y = "Modelo") +
    scale_x_continuous(
      breaks = seq(0, 1, by = 0.1),
      expand = expansion(mult = c(0, 0.1))
    ) +
    facet_wrap(~ Familia, ncol = 1, scales = "free_y") +
    theme_minimal(base_size = 13) +
    theme(
      panel.grid.major = element_line(color = "grey70", linewidth = 0.6),
      panel.grid.minor = element_line(color = "grey80", linewidth = 0.4),
      legend.position = "none",
      strip.text = element_text(size = 13, face = "bold"),
      axis.text.y = element_text(size = 7),
      axis.text.x = element_text(size = 10)
    )
  g
  # Exportar gráfico
  ggsave("graf_F1_fam.png", plot = g,
         width = 10, height = 20, units = "cm", dpi = 300)
}

#### Tabela de resultados ----------------------------------------
{#### Tabela de resultados ----------------------------------------
  install.packages("webshot2")
  install.packages("htmlwidgets")
  {
    # --- preparar tabela base ----------------------------------------------------
    tab_metricas <- resultados_df |>
      select(
        Modelo = Mod,
        TA, EPMP, MD, MDD, NC, TF, Vero,
        MC_Pre, MC_Rec, MC_F1, MC_MCC
      ) |>
      mutate(across(.cols = -Modelo, ~ round(.x, 4))) |>
      arrange(desc(Modelo))
    
    tab <- tab_metricas
    
    # --- parâmetros editáveis ----------------------------------------------------
    criterio <- c(
      TA     = "maior",
      EPMP   = "menor",    # <-- AGORA INCLUÍDO
      MD     = "menor",
      MDD    = "menor",
      NC     = "menor",
      Vero   = "maior",
      MC_Pre = "maior",
      MC_Rec = "maior",
      MC_F1  = "maior",
      MC_MCC = "maior"
    )
    
    cores_pos <- c("dodgerblue4", "dodgerblue1", "darkslategray3", "lightblue", "lightcyan3")
    
    # --- função para gerar vetores de cor ----------------------------------------
    cores_para_metrica <- function(v, tipo, cores_top = cores_pos, n_top = 5) {
      if (all(is.na(v))) return(rep("transparent", length(v)))
      
      decreasing <- (tipo == "maior")
      ordem <- order(v, decreasing = decreasing, na.last = TRUE)
      
      n_top <- min(n_top, length(v))
      top_idx <- ordem[seq_len(n_top)]
      
      cores <- rep("transparent", length(v))
      cores[top_idx] <- cores_top[seq_len(n_top)]
      cores[is.na(v)] <- "transparent"
      
      cores
    }
    
    # --- gerar lista de vetores de cor para todas as métricas --------------------
    lista_cores <- lapply(names(criterio), function(m) {
      cores_para_metrica(tab[[m]], criterio[m])
    })
    names(lista_cores) <- names(criterio)
    
    # --- construir tabela kable --------------------------------------------------
    k <- kbl(
      tab,
      align = "c",
      booktabs = TRUE,
      caption = "Top/Bottom por métrica",
      linesep = ""
    ) |>
      kable_classic(full_width = FALSE, html_font = "Arial") |>
      kable_paper("hover", "striped")
    
    # linhas verticais entre todas as colunas
    k <- k |>
      add_header_above(rep("", ncol(tab)), line = TRUE) |>
      column_spec(1:ncol(tab), border_left = TRUE, border_right = TRUE)
    
    # aplicar cores às colunas métricas
    for (m in names(lista_cores)) {
      col_index <- which(names(tab) == m)
      k <- k |> column_spec(col_index, background = lista_cores[[m]])
    }
    
    # adicionar linhas horizontais fortes
    k <- k |>
      add_header_above(c(" " = ncol(tab)), bold = TRUE, underline = TRUE) |>
      row_spec(0, extra_css = "border-bottom: 2px solid black;") |>
      row_spec(nrow(tab), extra_css = "border-bottom: 2px solid black;")
    
    k
  }
  
  library(kableExtra)
  library(htmltools)
  library(htmlwidgets)
  library(webshot2)
  
  salvar_tabela_imagem <- function(k, nome = "tabela_metricas") {
    
    library(htmltools)
    library(webshot2)
    
    # Criar nome do arquivo HTML
    html_file <- paste0(nome, ".html")
    
    # Criar página HTML mínima contendo sua tabela
    pagina <- htmltools::tagList(
      htmltools::tags$html(
        htmltools::tags$head(),
        htmltools::tags$body(
          htmltools::HTML(k)
        )
      )
    )
    
    # Salvar HTML
    htmltools::save_html(pagina, file = html_file)
    
    # Gerar PNG
    png_file <- paste0(nome, ".png")
    webshot(html_file, file = png_file, vwidth = 710, vheight = 2000)
    
    message("Arquivos gerados:\n",
            " - ", html_file, "\n",
            " - ", png_file)
  }
  
  
  salvar_tabela_imagem(k, "tabela_metricas")
  
  
}

#### Testes de diferença de classe

library(FSA)        # para Dunn
library(rstatix)    # opcional para tidy results

dados <- tab_teste   # Apenas para clareza

#===========================
# 1. TESTES PARA CLASSE MEMÓRIA
#===========================



