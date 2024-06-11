library(readr)
library(dplyr)
library(rpart)
library(rpart.plot)
library(xgboost)
library(caret)
library(mlbench)
library(randomForest)

brasil <- as.data.frame(read_csv("br.csv"))

#brasil <- brasil[6506:8405,]


resultado <- matrix(NA,nrow(brasil),1); brasil <- cbind(brasil,resultado)

for (i in 1:nrow(brasil)) {
  if(brasil$mandante_Placar[i]> brasil$visitante_Placar[i]){brasil$resultado[i] <- "VM"}
  if(brasil$mandante_Placar[i]==brasil$visitante_Placar[i]){brasil$resultado[i] <- "EM"}
  if(brasil$mandante_Placar[i]< brasil$visitante_Placar[i]){brasil$resultado[i] <- "VV"}
}

brasil <- as.data.frame(brasil)
# View(brasil)

PV <- select(brasil,vencedor,mandante_Placar,visitante_Placar)
brasil <- select(brasil,-vencedor,-mandante_Placar,-visitante_Placar,-ID)



brasil0 <- brasil

brasil0 <- select(brasil0,-formacao_mandante,-formacao_visitante,-tecnico_mandante,-tecnico_visitante)

brasil0 <- subset(brasil0,ano==2016)


table(brasil0$resultado)

# xgboost

br <- brasil0


br$rodata             <-  as.factor(br$rodata) 
br$data               <-  as.factor(br$data)  
br$mandante           <-  as.factor(br$mandante) 
br$hora               <-  as.factor(br$hora) 
br$visitante          <-  as.factor(br$visitante) 
#br$formacao_mandante  <-  as.factor(br$formacao_mandante) 
#br$formacao_visitante <-  as.factor(br$formacao_visitante) 
#br$tecnico_mandante   <-  as.factor(br$tecnico_mandante) 
#br$tecnico_visitante  <-  as.factor(br$tecnico_visitante) 
br$arena              <-  as.factor(br$arena) 
br$mandante_Estado    <-  as.factor(br$mandante_Estado) 
br$visitante_Estado   <-  as.factor(br$visitante_Estado) 
br$resultado          <-  as.factor(br$resultado) 

set.seed(123)

nrow(br)

n70 <- round(nrow(br)*0.7,0)
table(br$ano)

train <- br[1:n70,]
teste <- br[(n70+1):nrow(br),]


matriz_treino <- model.matrix(resultado ~ . - 1, data = train)
matriz_teste  <- model.matrix(resultado ~ . - 1, data = teste)

alvo_treino <- as.numeric(train$resultado) - 1  # Ajustar classes para 0, 1, 2
alvo_teste  <- as.numeric(teste$resultado) - 1


#alvo_treino <- alvo_treino[-length(alvo_treino)]


modelo_xgboost <- xgboost(data = matriz_treino,
                          label = alvo_treino,
                          nrounds = 5,
                          objective = "multi:softmax",
                          num_class = 3)


previsoesXGB <- predict(modelo_xgboost, newdata = matriz_teste)


previsoes_classe <- max.col(previsoesXGB) - 1


precisaoXGB <- sum(previsoes_classe == as.numeric(alvo_teste) - 1) / length(alvo_teste)
cat("Precisão do modelo:", precisaoXGB, "\n")


# Defina a grade de hiperparâmetros que você deseja testar
param_grid <- expand.grid(
  nrounds = c(50, 100, 150),            # Número de iterações
  max_depth = c(3, 6, 9),               # Profundidade máxima da árvore
  eta = c(0.01, 0.1, 0.3),              # Taxa de aprendizado
  gamma = 0,                            # Controle de regularização
  colsample_bytree = 1,                 # Fração de colunas a serem amostradas durante a construção de cada árvore
  min_child_weight = 1,                 # Mínimo somatório de instâncias de peso (hessiano) necessário em um filho
  subsample = 1                         # Fração de observações a serem amostradas durante a construção de cada árvore
)

# Função de controle para o método xgbTree
ctrl <- trainControl(method = "cv", number = 5, search = "grid")

# Execute o grid search
grid_search <- train(
  x = matriz_treino,
  y = alvo_treino,
  method = "xgbTree",
  trControl = ctrl,
  tuneGrid = param_grid,
  verbose = FALSE
)

# Exiba os melhores hiperparâmetros encontrados
print(grid_search)

# Use os melhores hiperparâmetros para treinar o modelo final
modelo_final <- xgboost(
  data = matriz_treino,
  label = alvo_treino,
  nrounds = grid_search$bestTune$nrounds,
  max_depth = grid_search$bestTune$max_depth,
  eta = grid_search$bestTune$eta,
  gamma = grid_search$bestTune$gamma,
  colsample_bytree = grid_search$bestTune$colsample_bytree,
  min_child_weight = grid_search$bestTune$min_child_weight,
  subsample = grid_search$bestTune$subsample,
  objective = "multi:softmax",
  num_class = 3
)

# Faça previsões no conjunto de teste
previsoesXGB_final <- predict(modelo_final, newdata = matriz_teste)

# Avalie o desempenho do modelo final
precisaoXGB_final <- sum(previsoesXGB_final == as.numeric(alvo_teste)) / length(alvo_teste)
cat("Precisão do modelo final:", precisaoXGB_final, "\n")



