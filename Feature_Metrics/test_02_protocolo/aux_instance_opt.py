import numpy as np
import torch


class DirectInstanceSuppression:
    """
    Otimização direta de uma imagem multiespectral.

    Objetivo:
        reduzir g_1(img)

    enquanto procura preservar:
        g_2(img) ~= g_2(img_original)
        g_3(img) ~= g_3(img_original)

    As funções g_1, g_2 e g_3 recebem numpy.ndarray (H, W, C)
    e retornam um número real.

    Como as funções são NumPy e podem não ser diferenciáveis,
    o gradiente é estimado por SPSA.
    """

    def __init__(
        self,
        img_5b,
        g_1,
        g_2=None,
        g_3=None,
        lambda_g2=1.0,
        lambda_g3=1.0,
        lambda_img=0.0,
        device=None,
    ):
        self.img_original = np.asarray(img_5b, dtype=np.float32).copy()

        if self.img_original.ndim != 3:
            raise ValueError("img_5b deve possuir shape (H, W, C).")

        self.g_1 = g_1
        self.g_2 = g_2
        self.g_3 = g_3

        self.lambda_g2 = lambda_g2
        self.lambda_g3 = lambda_g3
        self.lambda_img = lambda_img

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = torch.device(device)

        self.x0 = torch.from_numpy(self.img_original).to(self.device)

        # Valores que queremos preservar.
        self.g1_initial = float(g_1(self.img_original))
        self.g2_initial = (
            float(g_2(self.img_original))
            if g_2 is not None else None
        )
        self.g3_initial = (
            float(g_3(self.img_original))
            if g_3 is not None else None
        )

        self.history = []


    def _loss_numpy(self, img):
        """
        Avalia a função objetivo em uma imagem NumPy.
        """

        g1 = float(self.g_1(img))

        loss = g1

        g2 = None
        g3 = None

        if self.g_2 is not None:
            g2 = float(self.g_2(img))
            loss += (
                self.lambda_g2
                * (g2 - self.g2_initial) ** 2
            )

        if self.g_3 is not None:
            g3 = float(self.g_3(img))
            loss += (
                self.lambda_g3
                * (g3 - self.g3_initial) ** 2
            )

        if self.lambda_img > 0:
            diff = img - self.img_original
            loss += self.lambda_img * np.mean(diff ** 2)

        return loss, g1, g2, g3


    @torch.no_grad()
    def fit(
        self,
        epochs=200,
        lr=1e-2,
        perturbation_size=1e-3,
        clamp_min=None,
        clamp_max=None,
        verbose=10,
    ):
        """
        Retorna
        -------
        img_5b_suppress : np.ndarray
            Imagem otimizada com o mesmo shape da imagem original.
        """

        x = self.x0.clone()

        for epoch in range(epochs):

            # ---------------------------------------------------------
            # Direção aleatória SPSA: cada elemento recebe -1 ou +1
            # ---------------------------------------------------------
            delta = torch.empty_like(x).bernoulli_(0.5)
            delta.mul_(2.0).sub_(1.0)

            x_plus = x + perturbation_size * delta
            x_minus = x - perturbation_size * delta

            if clamp_min is not None or clamp_max is not None:
                low = -torch.inf if clamp_min is None else clamp_min
                high = torch.inf if clamp_max is None else clamp_max

                x_plus.clamp_(low, high)
                x_minus.clamp_(low, high)

            # ---------------------------------------------------------
            # Avaliação das duas perturbações
            # ---------------------------------------------------------
            img_plus = x_plus.cpu().numpy()
            img_minus = x_minus.cpu().numpy()

            loss_plus, _, _, _ = self._loss_numpy(img_plus)
            loss_minus, _, _, _ = self._loss_numpy(img_minus)

            # ---------------------------------------------------------
            # Estimativa SPSA do gradiente
            # ---------------------------------------------------------
            derivative = (
                (loss_plus - loss_minus)
                / (2.0 * perturbation_size)
            )

            grad_estimate = derivative * delta

            # ---------------------------------------------------------
            # Gradient descent
            # ---------------------------------------------------------
            x -= lr * grad_estimate

            if clamp_min is not None or clamp_max is not None:
                low = -torch.inf if clamp_min is None else clamp_min
                high = torch.inf if clamp_max is None else clamp_max
                x.clamp_(low, high)

            # ---------------------------------------------------------
            # Monitoramento
            # ---------------------------------------------------------
            if epoch % verbose == 0 or epoch == epochs - 1:

                img_current = x.cpu().numpy()

                loss, g1, g2, g3 = self._loss_numpy(img_current)

                self.history.append({
                    "epoch": epoch,
                    "loss": loss,
                    "g1": g1,
                    "g2": g2,
                    "g3": g3,
                })

                print(
                    f"Epoch {epoch:4d} | "
                    f"Loss: {loss:.6f} | "
                    f"g1: {g1:.6f} | "
                    f"g2: {g2} | "
                    f"g3: {g3}"
                )

        img_5b_suppress = x.cpu().numpy().copy()

        return img_5b_suppress




"""
Otimização direta por instância (per-instance optimization) para perturbação
controlada de imagens multiespectrais, usando SPSA (Simultaneous Perturbation
Stochastic Approximation) para estimar gradientes de funções g_i em NumPy
arbitrárias e potencialmente não diferenciáveis.

A variável otimizada é a própria imagem (não uma rede neural). PyTorch é usado
apenas como mecanismo de estado + otimizador (Adam), não para autograd através
de g_1/g_2/g_3.

Autor: (implementação gerada para o usuário)
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Dict, Any

import numpy as np
import torch


# Tipo das funções escalares fornecidas pelo usuário.
ScalarImageFn = Callable[[np.ndarray], float]


@dataclass
class OptimizationRecord:
    """Um ponto do histórico de otimização."""
    step: int
    loss: float
    g1: float
    g2: Optional[float]
    g3: Optional[float]
    d_reg: float


class SPSAImagePerturbationOptimizer:
    """
    Otimizador por instância que perturba uma imagem multiespectral ``x`` para
    obter ``x'`` tal que:

        1. g_1(x') seja reduzido;
        2. g_2(x') permaneça próximo de g_2(x)  (opcional);
        3. g_3(x') permaneça próximo de g_3(x)  (opcional);
        4. x' permaneça próximo de x segundo D(x', x) = ||x' - x||^2 (opcional).

    Objetivo otimizado (minimização):

        L(x') = g_1(x')
                + lambda2 * (g_2(x') - g_2(x))^2      [se g_2 fornecida]
                + lambda3 * (g_3(x') - g_3(x))^2      [se g_3 fornecida]
                + lambda_img * D(x', x)               [se lambda_img > 0]

    Como g_1, g_2, g_3 são funções NumPy arbitrárias (não necessariamente
    diferenciáveis), o gradiente da parte não-diferenciável do objetivo é
    estimado via SPSA (2 * n_directions avaliações de função por passo,
    independente do número de pixels). O termo de regularização D(x', x),
    por ser definido analiticamente por nós, tem gradiente exato calculado
    via autograd do PyTorch e é somado ao gradiente estimado por SPSA.

    O passo de atualização em si é feito por um otimizador Adam do PyTorch,
    ao qual atribuímos manualmente o gradiente estimado (``x.grad = ghat``).

    Parameters
    ----------
    img_5b : np.ndarray
        Imagem original, shape (H, W, C) — ex.: (960, 1280, 5).
    g1 : ScalarImageFn
        Função que se deseja reduzir. Recebe np.ndarray (H, W, C) -> float.
    g2, g3 : ScalarImageFn, opcional
        Funções que devem permanecer aproximadamente constantes.
    lambda2, lambda3 : float
        Pesos das penalidades quadráticas para g2 e g3.
    lambda_img : float
        Peso do termo de regularização D(x', x) = média((x'-x)^2).
        Se 0.0, o termo é desativado (nenhum custo extra é computado).
    pixel_min, pixel_max : float, opcional
        Limites de clamp aplicados a cada pixel após cada passo. Se None,
        são inferidos automaticamente a partir do dtype/range da imagem
        original (heurística simples — ajuste conforme seu domínio).
    device : str
        'cpu' ou 'cuda'. Note que, como g1/g2/g3 exigem NumPy, cada
        avaliação delas implica uma sincronização/cópia GPU->CPU. Se as
        funções g_i forem rápidas e o gargalo for a cópia, prefira 'cpu'.
    n_directions : int
        Número de direções aleatórias (Rademacher) usadas por passo para
        estimar o gradiente via SPSA. Aumentar reduz a variância da
        estimativa ao custo de mais avaliações de g_i por passo
        (custo = 2 * n_directions, independente do número de pixels).
    c : float
        Magnitude inicial da perturbação SPSA (na escala da própria imagem).
    c_decay : float
        Expoente de decaimento de c ao longo dos passos: c_k = c / (k+1)**c_decay.
        Valor teórico usual em SPSA: ~0.101.
    lr : float
        Taxa de aprendizado do Adam.
    seed : int, opcional
        Semente para reprodutibilidade das direções aleatórias.
    dtype : torch.dtype
        Precisão usada internamente para a otimização (float32 recomendado).
    """

    def __init__(
        self,
        img_5b: np.ndarray,
        g1: ScalarImageFn,
        g2: Optional[ScalarImageFn] = None,
        g3: Optional[ScalarImageFn] = None,
        lambda2: float = 1.0,
        lambda3: float = 1.0,
        lambda_img: float = 0.0,
        pixel_min: Optional[float] = None,
        pixel_max: Optional[float] = None,
        device: str = "cpu",
        n_directions: int = 4,
        c: float = 0.01,
        c_decay: float = 0.101,
        lr: float = 1e-2,
        seed: Optional[int] = None,
        dtype: torch.dtype = torch.float32,
    ):
        if img_5b.ndim != 3:
            raise ValueError(f"img_5b deve ter shape (H, W, C); recebido {img_5b.shape}")

        self.original_shape = img_5b.shape
        self.original_dtype = img_5b.dtype

        self.g1 = g1
        self.g2 = g2
        self.g3 = g3
        self.lambda2 = lambda2
        self.lambda3 = lambda3
        self.lambda_img = lambda_img

        self.device = torch.device(device)
        self.torch_dtype = dtype
        self.n_directions = n_directions
        self.c = c
        self.c_decay = c_decay
        self.lr = lr

        if seed is not None:
            self._generator = torch.Generator(device=self.device)
            self._generator.manual_seed(seed)
        else:
            self._generator = None

        # Limites de pixel: se não fornecidos, inferidos do range da imagem original.
        if pixel_min is None:
            pixel_min = float(np.min(img_5b))
        if pixel_max is None:
            pixel_max = float(np.max(img_5b))
        self.pixel_min = pixel_min
        self.pixel_max = pixel_max

        # Imagem original como tensor "congelado" (referência), sem grad.
        self.x0 = torch.as_tensor(
            img_5b, dtype=self.torch_dtype, device=self.device
        ).clone()

        # Valores de referência de g2/g3 na imagem original.
        self.g2_ref = float(self.g2(img_5b)) if self.g2 is not None else None
        self.g3_ref = float(self.g3(img_5b)) if self.g3 is not None else None

        self.history: List[OptimizationRecord] = []

    # ------------------------------------------------------------------ #
    # Utilidades internas
    # ------------------------------------------------------------------ #

    def _to_numpy(self, x: torch.Tensor) -> np.ndarray:
        """Converte tensor para np.ndarray no shape/dtype originais, sem grad."""
        return (
            x.detach()
            .to("cpu")
            .numpy()
            .astype(self.original_dtype, copy=False)
        )

    def _clamp(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(x, self.pixel_min, self.pixel_max)

    def _rademacher_like(self, x: torch.Tensor) -> torch.Tensor:
        """Vetor aleatório com entradas em {-1, +1}, mesmo shape de x."""
        if self._generator is not None:
            rand = torch.randint(
                0, 2, x.shape, device=self.device, generator=self._generator
            )
        else:
            rand = torch.randint(0, 2, x.shape, device=self.device)
        return (rand.to(self.torch_dtype) * 2.0) - 1.0

    def _blackbox_loss(self, x: torch.Tensor) -> Dict[str, float]:
        """
        Avalia a parte NÃO diferenciável do objetivo (g1 + penalidades em
        g2/g3) para um dado tensor x, convertendo para NumPy apenas uma vez.
        Retorna também os valores individuais de g1/g2/g3 para o histórico.
        """
        x_np = self._to_numpy(x)

        g1_val = float(self.g1(x_np))
        loss = g1_val

        g2_val = None
        if self.g2 is not None:
            g2_val = float(self.g2(x_np))
            loss += self.lambda2 * (g2_val - self.g2_ref) ** 2

        g3_val = None
        if self.g3 is not None:
            g3_val = float(self.g3(x_np))
            loss += self.lambda3 * (g3_val - self.g3_ref) ** 2

        return {"loss": loss, "g1": g1_val, "g2": g2_val, "g3": g3_val}

    def _spsa_gradient(self, x: torch.Tensor, step: int) -> torch.Tensor:
        """
        Estima o gradiente da parte não-diferenciável do objetivo via SPSA,
        fazendo a média sobre `n_directions` direções aleatórias.

        Custo: 2 * n_directions avaliações de (g1, g2, g3), independente do
        número de pixels da imagem.
        """
        ck = self.c / ((step + 1) ** self.c_decay)
        grad_accum = torch.zeros_like(x)

        for _ in range(self.n_directions):
            delta = self._rademacher_like(x)  # entradas em {-1, +1}

            x_plus = self._clamp(x + ck * delta)
            x_minus = self._clamp(x - ck * delta)

            loss_plus = self._blackbox_loss(x_plus)["loss"]
            loss_minus = self._blackbox_loss(x_minus)["loss"]

            # Estimador SPSA: ghat_i = (L+ - L-) / (2*ck*delta_i)
            # Como delta_i in {-1,+1}, 1/delta_i == delta_i.
            grad_accum += (loss_plus - loss_minus) / (2.0 * ck) * delta

        grad_accum /= self.n_directions
        return grad_accum

    def _reg_term(self, x: torch.Tensor) -> torch.Tensor:
        """D(x', x) = MSE(x', x). Diferenciável -> gradiente exato via autograd."""
        return torch.mean((x - self.x0) ** 2)

    # ------------------------------------------------------------------ #
    # API pública
    # ------------------------------------------------------------------ #

    def fit(
        self,
        num_steps: int = 200,
        verbose: bool = True,
        log_every: int = 10,
    ) -> np.ndarray:
        """
        Executa a otimização por `num_steps` iterações.

        Returns
        -------
        np.ndarray
            Imagem perturbada (img_5b_suppress), mesmo shape e dtype da
            imagem original.
        """
        # Variável otimizada: cópia da imagem original, como "parâmetro" do Adam.
        x = self.x0.clone().detach().requires_grad_(True)
        optimizer = torch.optim.Adam([x], lr=self.lr)

        self.history = []

        for step in range(num_steps):
            optimizer.zero_grad()

            # --- Parte 1: gradiente exato do termo de regularização (se houver) ---
            if self.lambda_img > 0.0:
                reg_loss = self.lambda_img * self._reg_term(x)
                reg_loss.backward()  # popula x.grad com o gradiente exato de D
                exact_grad = x.grad.detach().clone()
                d_reg_value = float(reg_loss.item())
            else:
                exact_grad = torch.zeros_like(x)
                d_reg_value = 0.0

            # --- Parte 2: gradiente estimado (SPSA) da parte não-diferenciável ---
            x_detached = x.detach()
            spsa_grad = self._spsa_gradient(x_detached, step)

            # --- Combina os dois gradientes e aplica o passo do Adam ---
            optimizer.zero_grad()
            x.grad = exact_grad + spsa_grad
            optimizer.step()

            # Projeta de volta para o domínio válido de pixels.
            with torch.no_grad():
                x.copy_(self._clamp(x))

            # --- Logging (avaliação no ponto atualizado, para o histórico) ---
            with torch.no_grad():
                metrics = self._blackbox_loss(x)
                total_loss = metrics["loss"] + d_reg_value

            self.history.append(
                OptimizationRecord(
                    step=step,
                    loss=total_loss,
                    g1=metrics["g1"],
                    g2=metrics["g2"],
                    g3=metrics["g3"],
                    d_reg=d_reg_value,
                )
            )

            if verbose and (step % log_every == 0 or step == num_steps - 1):
                g2_str = f"{metrics['g2']:.4f}" if metrics["g2"] is not None else "N/A"
                g3_str = f"{metrics['g3']:.4f}" if metrics["g3"] is not None else "N/A"
                print(
                    f"[step {step:4d}] loss={total_loss:.6f} "
                    f"g1={metrics['g1']:.6f} g2={g2_str} g3={g3_str} "
                    f"D={d_reg_value:.6f}"
                )

        return self._to_numpy(x)

    def get_history_dict(self) -> Dict[str, List[Any]]:
        """Retorna o histórico como dict de listas (fácil de plotar/salvar)."""
        return {
            "step": [r.step for r in self.history],
            "loss": [r.loss for r in self.history],
            "g1": [r.g1 for r in self.history],
            "g2": [r.g2 for r in self.history],
            "g3": [r.g3 for r in self.history],
            "d_reg": [r.d_reg for r in self.history],
        }


# ---------------------------------------------------------------------- #
# Exemplo mínimo de uso
# ---------------------------------------------------------------------- #
if __name__ == "__main__":
    # Imagem multiespectral sintética (960 x 1280 x 5), valores em [0, 1].
    rng = np.random.default_rng(0)
    img_5b = rng.random((960, 1280, 5)).astype(np.float32)

    # Exemplos de funções escalares arbitrárias em NumPy (não-diferenciáveis
    # de propósito: usam np.median, que não tem gradiente bem definido).
    def g_1(img: np.ndarray) -> float:
        """Ex.: 'força' de um sinal indesejado no canal 0."""
        return float(np.median(img[:, :, 0]))

    def g_2(img: np.ndarray) -> float:
        """Ex.: brilho médio geral — deve ser preservado."""
        return float(img.mean())

    def g_3(img: np.ndarray) -> float:
        """Ex.: contraste (desvio padrão) do canal 3 — deve ser preservado."""
        return float(img[:, :, 3].std())

    optimizer = SPSAImagePerturbationOptimizer(
        img_5b=img_5b,
        g1=g_1,
        g2=g_2,
        g3=g_3,
        lambda2=50.0,
        lambda3=50.0,
        lambda_img=0.1,
        pixel_min=0.0,
        pixel_max=1.0,
        device="cpu",
        n_directions=4,
        c=0.02,
        lr=5e-3,
        seed=42,
    )

    img_5b_suppress = optimizer.fit(num_steps=100, verbose=True, log_every=10)

    print("\nShape de entrada :", img_5b.shape, img_5b.dtype)
    print("Shape de saída   :", img_5b_suppress.shape, img_5b_suppress.dtype)
    print("g1 original ->", g_1(img_5b), " | g1 final ->", g_1(img_5b_suppress))
    print("g2 original ->", g_2(img_5b), " | g2 final ->", g_2(img_5b_suppress))
    print("g3 original ->", g_3(img_5b), " | g3 final ->", g_3(img_5b_suppress))


    plot_rgb(img_5b)
    plot_rgb(img_5b_suppress)

