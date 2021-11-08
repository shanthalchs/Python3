def zipFila(fila, p = 4):
    tam = 16
    row = []
    for n in range(0,len(fila),tam * p):
        bloques = []
        for m in range(0, tam, p):
            subBloque = []
            for i in range(p):
                inicio = m + (tam*i) + n
                fin = inicio + p
                subBloque += list(fila[inicio:fin])
            bloques += [np.mean(subBloque)]
            
        row += bloques
        
    return row

def zipData(datos,p = 4):
    datosComprimidos = []
    for n in range(datos.shape[0]):
        datosComprimidos += [zipFila(datos.iloc[n,:],p)]
    
    return pd.DataFrame(datosComprimidos)