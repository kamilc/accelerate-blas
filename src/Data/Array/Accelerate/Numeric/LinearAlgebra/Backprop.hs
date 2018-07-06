{-# LANGUAGE TypeSynonymInstances #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# OPTIONS_GHC -fno-warn-orphans                     #-}

-- |
-- Module      : Data.Array.Accelerate.Numeric.LinearAlgebra.Backprop
-- Copyright   : [2018] Kamil Ciemniewski
-- License     : BSD3
--
-- Maintainer  : Kamil Ciemniewski <kamil@ciemniew.ski>
-- Stability   : experimental
-- Portability : non-portable (GHC extensions)
--

module Data.Array.Accelerate.Numeric.LinearAlgebra.Backprop where

import qualified Data.Array.Accelerate as A
import qualified Data.Array.Accelerate.Numeric.LinearAlgebra as L
import qualified Data.Array.Accelerate.Numeric.LinearAlgebra.BLAS.Level1 as L1
import qualified Data.Array.Accelerate.Numeric.LinearAlgebra.BLAS.Level2 as L2
import Numeric.Backprop

type Scalar n = A.Acc (L.Scalar n)

type Vector n = A.Acc (L.Vector n)

type Matrix n = A.Acc (L.Matrix n)

instance (A.Shape sh, A.Elt n, Num n, Num (A.Exp n)) => Backprop (A.Acc (A.Array sh n)) where
  zero = A.map (\_ -> 0)
  one = A.map (\_ -> 1)
  add = A.zipWith (+)

-- | Vector - vector product
dotu ::
  (Reifies s W, L1.Numeric e, Num e)
  => BVar s (Vector e)
  -> BVar s (Vector e)
  -> BVar s (Scalar e)
dotu =
  liftOp2 . op2 $ \x y ->
    ( L1.dotu x y
    , \dzdy ->
        let d = A.the dzdy
         in (A.map (\a -> a * d) y, A.map (\a -> a * d) x))

(<.>) ::
  (Reifies s W, L1.Numeric e, Num e)
  => BVar s (Vector e)
  -> BVar s (Vector e)
  -> BVar s (Scalar e)
(<.>) = dotu
infixr 8 <.>
{-# INLINE (<.>) #-}

-- | Outer vector - vector product
outer ::
  forall s e. (Reifies s W, L1.Numeric e, Num e)
  => BVar s (Vector e)
  -> BVar s (Vector e)
  -> BVar s (Matrix e)
outer =
  liftOp2 . op2 $ \x y ->
    ( (L.><) x y
    , \dzdy ->
        ((dzdy L.#> y), (L2.gemv (A.constant 1) L2.T dzdy x))
      )

(><) ::
  forall s e. (Reifies s W, L1.Numeric e, Num e)
  => BVar s (Vector e)
  -> BVar s (Vector e)
  -> BVar s (Matrix e)
(><) = outer
infixr 8 ><
{-# INLINE (><) #-}
