{-# LANGUAGE TypeSynonymInstances #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE AllowAmbiguousTypes #-}
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
import Numeric.Backprop

type Scalar n = A.Acc (L.Scalar n)

type Vector n = A.Acc (L.Vector n)

type Matrix n = A.Acc (L.Matrix n)

instance (A.Elt n, Num n, Num (A.Exp n)) => Backprop (Scalar n) where
  zero s = A.fill (A.shape s) (A.constant 0)
  one s = A.fill (A.shape s) (A.constant 1)
  add l r = A.unit $ (A.the l) + (A.the r)

instance (A.Elt n, Num n, Num (A.Exp n)) => Backprop (Vector n) where
  zero s = A.fill (A.shape s) (A.constant 0)
  one s = A.fill (A.shape s) (A.constant 1)
  add l r = A.zipWith (+) l r

instance (A.Elt n, Num n, Num (A.Exp n)) => Backprop (Matrix n) where
  zero s = A.fill (A.shape s) (A.constant 0)
  one s = A.fill (A.shape s) (A.constant 1)
  add l r = A.zipWith (+) l r

(<.>) ::
  (Reifies s W, Num (A.Exp n), L1.Numeric e, Num e)
  => BVar s (Vector e)
  -> BVar s (Vector e)
  -> BVar s (Scalar e)
(<.>) =
  liftOp2 . op2 $ \x y ->
    ( L1.dotu x y
    , \dzdy ->
        let d = A.the dzdy
         in (A.map (\a -> a * d) y, A.map (\a -> a * d) x))
