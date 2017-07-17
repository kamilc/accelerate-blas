
module Hedgehog.Gen.Array where

import Data.Array.Accelerate                              as A
import Prelude                                            as P

import Hedgehog                                           ( Gen )
import qualified Hedgehog.Gen                             as Gen
import qualified Hedgehog.Range                           as Range


-- Generate an array of the given shape
--
genArray
    :: (Shape sh, Elt e)
    => sh
    -> Gen e
    -> Gen (Array sh e)
genArray sh gen =
  fromList sh <$> Gen.list (Range.singleton (arraySize sh)) gen

