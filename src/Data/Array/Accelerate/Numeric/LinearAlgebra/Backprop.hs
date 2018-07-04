{-# LANGUAGE TypeSynonymInstances                     #-}
{-# LANGUAGE FlexibleInstances                        #-}
{-# LANGUAGE FlexibleContexts                         #-}
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

import           Numeric.Backprop
import qualified Data.Array.Accelerate.Numeric.LinearAlgebra as L
import qualified Data.Array.Accelerate as A

instance (A.Elt n, Num n, Num (A.Exp n)) => Backprop (A.Acc (L.Scalar n)) where
  zero s = A.fill (A.shape s) (A.constant 0)
  one s = A.fill (A.shape s) (A.constant 1)
  add l r = A.unit $ (A.the l) + (A.the r)

instance (A.Elt n, Num n, Num (A.Exp n)) => Backprop (A.Acc (L.Vector n)) where
  zero s = A.fill (A.shape s) (A.constant 0)
  one s = A.fill (A.shape s) (A.constant 1)
  add l r = A.zipWith (+) l r

instance (A.Elt n, Num n, Num (A.Exp n)) => Backprop (A.Acc (L.Matrix n)) where
  zero s = A.fill (A.shape s) (A.constant 0)
  one s = A.fill (A.shape s) (A.constant 1)
  add l r = A.zipWith (+) l r
