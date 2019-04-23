import bf_add_stuff
import bifrost as bf

a = bf.ndarray([1,2,3,4,5,6,7,8,9,10],  dtype='f32')
b = bf.ndarray([2,3,4,5,6,7,8,9,10,11], dtype='f32')

import bifrost as bf

a = bf.ndarray([1,2,3,4,5,6,7,8,9,10],  dtype='f32')
b = bf.ndarray([2,3,4,5,6,7,8,9,10,11], dtype='f32')

bf_add_stuff.AddStuff(a.as_BFarray(), b.as_BFarray())
