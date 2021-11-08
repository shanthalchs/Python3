class consensoPropio:
    def __init__(self,datos,porcentajentrenamiento): 
        self.__datos = datos
        self.__porcentajentrenamiento = porcentajentrenamiento
      
    @property
    def datos(self):
        return self.__datos
    @datos.setter
    def datos(self, datos):
        self.__datos = datos  
        
    @property
    def porcentajentrenamiento(self):
        return self.__porcentajentrenamiento
    @porcentajentrenamiento.setter
    def porcentaje_entrenamiento(self, porcentajentrenamiento):
        self.__porcentajentrenamiento = porcentajentrenamiento    
        
    @property
    def train(self):
        return self.__train
    @train.setter
    def train(self, train):
        self.__train = train
        
    @property
    def test(self):
        return self.__test
    @test.setter
    def test(self, test):
        self.__test = test 
        
    @property
    def sample_size(self):
        return self.__sample_size
    @sample_size.setter
    def sample_size(self, sample_size):
        self.__sample_size = sample_size 
        
    @property
    def modelos(self):
        return self.__modelos
    @modelos.setter
    def modelos(self, modelos):
        self.__modelos = modelos 
        
    @property
    def precisiones(self):
        return self.__precisiones
    @precisiones.setter
    def precisiones(self, precisiones):
        self.__precisiones = precisiones 
   
    def split(self):
        self.train,self.test = train_test_split(self.datos,train_size=self.porcentajentrenamiento, random_state=0)
        self.sample_size = 3000
        
    def fit(self):
        precisionesglobales = []
        modelo = []
        modelos = []
        
        ## KNN
        boot = resample(self.train, replace=True, n_samples=self.sample_size, random_state=1)
        columns = boot.shape[1]
        X_aux = boot.iloc[:,0:(columns-1)] 
        y_aux = boot.iloc[:,(columns-1):columns] 
        X_train, X_test, y_train, y_test = train_test_split(X_aux, y_aux, train_size=self.porcentajentrenamiento, random_state=0)
        instancia_knn = KNeighborsClassifier(n_neighbors=3)
        instancia_knn.fit(X_train,y_train.iloc[:,0].values)
        modelo.append(instancia_knn)
        prediccion = instancia_knn.predict(X_test)
        matriz = confusion_matrix(y_test, prediccion)
        precisionglobal = (matriz[0][0]+matriz[1][1])/(matriz[0][0]+matriz[0][1]+matriz[1][0]+matriz[1][1]) 
        precisionesglobales.append(round(precisionglobal, 4))    
        
        
        ## DECISIONTREE
        boot = resample(self.train, replace=True, n_samples=self.sample_size, random_state=1)
        columns = boot.shape[1]
        X_aux = boot.iloc[:,0:(columns-1)] 
        y_aux = boot.iloc[:,(columns-1):columns] 
        X_train, X_test, y_train, y_test = train_test_split(X_aux, y_aux, train_size=self.porcentajentrenamiento, random_state=0)
        instancia_arbol = DecisionTreeClassifier(random_state=0)
        instancia_arbol.fit(X_train,y_train)
        modelo.append(instancia_arbol)
        prediccion = instancia_arbol.predict(X_test)
        matriz = confusion_matrix(y_test, prediccion)
        precisionglobal = (matriz[0][0]+matriz[1][1])/(matriz[0][0]+matriz[0][1]+matriz[1][0]+matriz[1][1]) 
        precisionesglobales.append(round(precisionglobal, 4)) 
        
        
        ## ADA BOOSTING
        boot = resample(self.train, replace=True, n_samples=self.sample_size, random_state=1)
        columns = boot.shape[1]
        X_aux = boot.iloc[:,0:(columns-1)] 
        y_aux = boot.iloc[:,(columns-1):columns] 
        X_train, X_test, y_train, y_test = train_test_split(X_aux, y_aux, train_size=self.porcentajentrenamiento, random_state=0)
        instancia_potenciacion = AdaBoostClassifier(n_estimators=10, random_state=0)
        instancia_potenciacion.fit(X_train,y_train.iloc[:,0].values)
        modelo.append(instancia_potenciacion)
        prediccion = instancia_potenciacion.predict(X_test)
        matriz = confusion_matrix(y_test, prediccion)
        precisionglobal = (matriz[0][0]+matriz[1][1])/(matriz[0][0]+matriz[0][1]+matriz[1][0]+matriz[1][1]) 
        precisionesglobales.append(round(precisionglobal, 4)) 
        
        
        ##XG BOOSTING
        boot = resample(self.train, replace=True, n_samples=self.sample_size, random_state=1)
        columns = boot.shape[1]
        X_aux = boot.iloc[:,0:(columns-1)] 
        y_aux = boot.iloc[:,(columns-1):columns] 
        X_train, X_test, y_train, y_test = train_test_split(X_aux, y_aux, train_size=self.porcentajentrenamiento, random_state=0)
        instancia_potenciacion = GradientBoostingClassifier(n_estimators=10, random_state=0)
        instancia_potenciacion.fit(X_train,y_train.iloc[:,0].values)
        modelo.append(instancia_potenciacion)
        prediccion = instancia_potenciacion.predict(X_test)
        matriz = confusion_matrix(y_test, prediccion)
        precisionglobal = (matriz[0][0]+matriz[1][1])/(matriz[0][0]+matriz[0][1]+matriz[1][0]+matriz[1][1]) 
        precisionesglobales.append(round(precisionglobal, 4))
        
        modelos.append(modelo)
        modelos.append(precisionesglobales)
        self.modelos = modelos
        
        
    def predict(self,vector):
        columns = self.test.shape[1]
        X_aux = self.test.iloc[:,0:(columns-1)] 
        y_aux = self.test.iloc[:,(columns-1):columns]
        knn = self.modelos[0][0]
        prediccion_knn = knn.predict_proba(X_aux)
        print("KNN")
        print(prediccion_knn)
        randomtree = self.modelos[0][1]
        prediccion_randomtree = randomtree.predict_proba(X_aux)
        print("Random Tree")
        print(prediccion_randomtree)
        ada = self.modelos[0][2]
        prediccion_ada = ada.predict_proba(X_aux)
        print("ADA Boosting")
        print(prediccion_ada)
        xg = self.modelos[0][3]
        prediccion_xg = xg.predict_proba(X_aux)
        print("XG Boosting")
        print(prediccion_xg)
         ## Se multiplica la probabilidad contra la precisión global para darle mayor peso a los modelos con mejor precisión global
        prediccion_knn =  self.modelos[1][0] * prediccion_knn
        prediccion_randomtree = self.modelos[1][1] * prediccion_randomtree
        prediccion_ada = self.modelos[1][2] * prediccion_ada  
        prediccion_xg = self.modelos[1][3] * prediccion_xg
        matriz = []
        # Se suman las 4 probabilidades y se divide entre 4 para obtener la probabilidad ponderada
        for i in range(prediccion_knn.shape[0]):
            fila = []
            for j in range(prediccion_knn.shape[1]):
                prediccion = (prediccion_knn[i][j]+prediccion_randomtree[i][j]+prediccion_randomtree[i][j]+prediccion_ada[i][j])/4
                fila.append(prediccion)
            matriz.append(fila)
        # El valor que tenga más probabilidades es el que se devolverá
        predicciones = []
        for i in range(prediccion_knn.shape[0]):
            indice = matriz[i].index(max(matriz[i]))
            predicciones.append(vector[indice])
        self.prediccion = predicciones
        
    def precisiones(self):
        columns = self.test.shape[1]
        y_aux = self.test.iloc[:,(columns-1):columns]
        MC = confusion_matrix(y_aux,self.prediccion )
        nombres = list(np.unique(y_aux))
        precision_global = np.sum(MC.diagonal()) / np.sum(MC)
        error_global = 1 - precision_global
        precision_categoria  = pd.DataFrame(MC.diagonal()/np.sum(MC,axis = 1)).T
        if nombres!=None:
            precision_categoria.columns = nombres
            indices = {"Matriz de Confusión":MC, 
            "Precisión Global":precision_global, 
            "Error Global":error_global, 
            "Precisión por categoría":precision_categoria}
        for k in indices:
            print("\n%s:\n%s"%(k,str(indices[k])))
        return MC
