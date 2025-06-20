import torch.nn.functional as F
import torch

def latentes_explicitas(image_tensors, block_sizes, p_scrambles): # te, [0,2,3] p_scr[0,0.5]
    b = image_tensors.shape[0]
    #print(b)
    main_tensor = block_scramble_v2(image_tensors[0], block_sizes[0], p_scrambles[0])
    for i in range(1,b):
        acum = recorte_aleatorio(image_tensors[i],block_sizes[i]*2)
        acum = block_scramble_v2(acum, block_sizes[i], p_scrambles[i])
        main_tensor = colocar_tensor_random(main_tensor, acum)

    
    return main_tensor


def block_scramble(image_tensor, block_size):
    if image_tensor.ndim == 3:
        image_tensor = image_tensor.unsqueeze(0)

    b, c, h, w = image_tensor.shape
    
    if h % block_size != 0 or w % block_size != 0:
        raise ValueError("La altura y la anchura de la imagen deben ser divisibles por el tamaño del bloque.")

    # 2. Extraer los bloques usando F.unfold
    # kernel_size es el tamaño de nuestro bloque.
    # stride es cuánto se mueve la ventana. Al hacerlo igual al tamaño del bloque,
    # los bloques no se superponen.
    # El resultado 'patches' tiene la forma (B, C * block_h * block_w, num_blocks)
    patches = F.unfold(image_tensor, kernel_size=block_size, stride=block_size)
    print(patches.size())
    # 3. Generar y aplicar la permutación a los bloques
    # 'patches.shape[-1]' es el número total de bloques.
    num_blocks = patches.shape[-1]
    
    # Generamos una permutación aleatoria que servirá como clave.
    permutation_key = torch.randperm(num_blocks, device=image_tensor.device)
    print(permutation_key.size())
    # Barajamos los bloques (la última dimensión de 'patches') usando la clave.
    # Para batch_size > 1, se aplicaría la misma permutación a cada imagen del lote.
    scrambled_patches = patches[:, :, permutation_key] #torch.Size([1, 16, 9])
    print(scrambled_patches.size())
    # 4. Reconstruir la imagen con los bloques mezclados usando F.fold
    # F.fold hace la operación inversa a F.unfold.
    scrambled_image = F.fold(
        scrambled_patches,
        output_size=(h, w),
        kernel_size=(block_size, block_size),
        stride=(block_size, block_size)
    )

    return scrambled_image

def block_scramble_v2(image_tensor, block_size, p_scramble=0.0):
    if image_tensor.ndim == 3:
        image_tensor = image_tensor.unsqueeze(0)
    b, c, h, w = image_tensor.shape
    
    if h % block_size != 0 or w % block_size != 0:
        raise ValueError("La altura y la anchura de la imagen deben ser divisibles por el tamaño del bloque.")

    patches = F.unfold(image_tensor, kernel_size=block_size, stride=block_size)
    
    # --- Nuevo bloque de código para barajar internamente los pixeles de cada parche ---
    
    # Remodelar para aislar los pixeles de cada bloque: (B, C, K*K, L)
    # donde K es block_size y L es el número de bloques.
    num_blocks = patches.shape[-1]
    pixels_per_block = block_size * block_size
    reshaped_patches = patches.view(b, c, pixels_per_block, num_blocks)

    for i in range(num_blocks):
        if torch.rand(1).item() < p_scramble:
            # Generar una permutación para los pixeles DENTRO de este bloque
            internal_permutation = torch.randperm(pixels_per_block, device=image_tensor.device)
            
            # Aplicar la permutación al i-ésimo bloque en todos los canales y lotes
            reshaped_patches[:, :, :, i] = reshaped_patches[:, :, internal_permutation, i]
            
    # Volver a la forma original de 'patches' para el siguiente paso
    patches_internally_scrambled = reshaped_patches.view(b, c * pixels_per_block, num_blocks)
    
    # --- Fin del nuevo bloque de código ---

    # Generar una permutación para las POSICIONES de los bloques
    location_permutation_key = torch.randperm(num_blocks, device=image_tensor.device)
    
    # Barajar las posiciones de los bloques (ya internamente barajados o no)
    scrambled_patches = patches_internally_scrambled[:, :, location_permutation_key]

    # Reconstruir la imagen final
    scrambled_image = F.fold(
        scrambled_patches,
        output_size=(h, w),
        kernel_size=(block_size, block_size),
        stride=(block_size, block_size)
    )

    return scrambled_image

def recorte_aleatorio(tensor_entrada, tamano_salida):
    if tensor_entrada.ndim == 3:
        tensor_entrada = tensor_entrada.unsqueeze(0)

    if isinstance(tamano_salida, int):
        tamano_salida = (tamano_salida, tamano_salida)
        
    h_salida, w_salida = tamano_salida
    b, c, h_entrada, w_entrada = tensor_entrada.shape

    if h_salida > h_entrada or w_salida > w_entrada:
        raise ValueError("El tamaño de salida no puede ser mayor que el tensor de entrada.")

    rango_h = h_entrada - h_salida
    rango_w = w_entrada - w_salida

    top = torch.randint(0, rango_h + 1, size=(1,)).item()
    left = torch.randint(0, rango_w + 1, size=(1,)).item()
    
    tensor_recortado = tensor_entrada[:, :, top:top + h_salida, left:left + w_salida]
    
    return tensor_recortado

def colocar_tensor_random(tensor_grande, tensor_pequeno):
    """
    Coloca un tensor pequeño (B, C, h, w) dentro de uno grande (B, C, H, W)
    en una posición espacial aleatoria.
    """
    # Se crea una copia para no modificar el tensor original
    resultado = tensor_grande.clone()
    
    # --- Comprobaciones de seguridad ---
    
    # 1. Validar que las dimensiones de batch y canal coincidan
    if tensor_grande.shape[0] != tensor_pequeno.shape[0] or \
       tensor_grande.shape[1] != tensor_pequeno.shape[1]:
        raise ValueError(f"Las dimensiones de batch y canal no coinciden. "
                         f"Grande: {tensor_grande.shape[:2]}, Pequeño: {tensor_pequeno.shape[:2]}")

    # 2. Validar que el tensor pequeño quepa en el grande
    if tensor_pequeno.shape[2] > tensor_grande.shape[2] or \
       tensor_pequeno.shape[3] > tensor_grande.shape[3]:
        raise ValueError("El tensor pequeño es espacialmente más grande que el tensor grande.")

    # --- Lógica principal ---
    
    # Obtener las dimensiones espaciales
    _, _, h_grande, w_grande = tensor_grande.shape
    _, _, h_pequeno, w_pequeno = tensor_pequeno.shape

    # Calcular el rango máximo para la posición inicial
    max_fila = h_grande - h_pequeno
    max_col = w_grande - w_pequeno
    
    # Elegir una posición inicial aleatoria
    fila_inicio = torch.randint(0, max_fila + 1, size=(1,)).item()
    col_inicio = torch.randint(0, max_col + 1, size=(1,)).item()
    
    # --- CORRECCIÓN CLAVE ---
    # Se especifica [:, :] al principio para indicar que la operación
    # se aplica a todas las dimensiones de batch y canal.
    resultado[:, :, fila_inicio : fila_inicio + h_pequeno, col_inicio : col_inicio + w_pequeno] = tensor_pequeno
    
    return resultado