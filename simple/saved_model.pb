µ
¾
:
Add
x"T
y"T
z"T"
Ttype:
2	
W
AddN
inputs"T*N
sum"T"
Nint(0"!
Ttype:
2	
ī
	ApplyAdam
var"T	
m"T	
v"T
beta1_power"T
beta2_power"T
lr"T

beta1"T

beta2"T
epsilon"T	
grad"T
out"T" 
Ttype:
2	"
use_lockingbool( "
use_nesterovbool( 
x
Assign
ref"T

value"T

output_ref"T"	
Ttype"
validate_shapebool("
use_lockingbool(
s
	AssignAdd
ref"T

value"T

output_ref"T" 
Ttype:
2	"
use_lockingbool( 
R
BroadcastGradientArgs
s0"T
s1"T
r0"T
r1"T"
Ttype0:
2	
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
8
Const
output"dtype"
valuetensor"
dtypetype
B
Equal
x"T
y"T
z
"
Ttype:
2	

,
Exp
x"T
y"T"
Ttype:

2
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
,
Floor
x"T
y"T"
Ttype:
2
?
FloorDiv
x"T
y"T
z"T"
Ttype:
2	
=
Greater
x"T
y"T
z
"
Ttype:
2	
B
GreaterEqual
x"T
y"T
z
"
Ttype:
2	
.
Identity

input"T
output"T"	
Ttype
.
Log1p
x"T
y"T"
Ttype:

2
$

LogicalAnd
x

y

z

p
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:
	2
;
Maximum
x"T
y"T
z"T"
Ttype:

2	

Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(
=
Mul
x"T
y"T
z"T"
Ttype:
2	
.
Neg
x"T
y"T"
Ttype:

2	

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:

Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
~
RandomUniform

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	
>
RealDiv
x"T
y"T
z"T"
Ttype:
2	
5

Reciprocal
x"T
y"T"
Ttype:

2	
E
Relu
features"T
activations"T"
Ttype:
2	
V
ReluGrad
	gradients"T
features"T
	backprops"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
0
Round
x"T
y"T"
Ttype:

2	
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
:
Sub
x"T
y"T
z"T"
Ttype:
2	

Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
c
Tile

input"T
	multiples"
Tmultiples
output"T"	
Ttype"

Tmultiplestype0:
2	
s

VariableV2
ref"dtype"
shapeshape"
dtypetype"
	containerstring "
shared_namestring 
&
	ZerosLike
x"T
y"T"	
Ttype"serve*1.11.02b'v1.11.0-rc2-4-gc19e29306c'øū
l
	inputs_phPlaceholder*
dtype0*'
_output_shapes
:’’’’’’’’’	*
shape:’’’’’’’’’	
l
	labels_phPlaceholder*
dtype0*'
_output_shapes
:’’’’’’’’’*
shape:’’’’’’’’’
e
random_uniform/shapeConst*
valueB"	   x   *
dtype0*
_output_shapes
:
W
random_uniform/minConst*
valueB
 *n×\¾*
dtype0*
_output_shapes
: 
W
random_uniform/maxConst*
valueB
 *n×\>*
dtype0*
_output_shapes
: 

random_uniform/RandomUniformRandomUniformrandom_uniform/shape*
T0*
dtype0*
_output_shapes

:	x*
seed2 *

seed 
b
random_uniform/subSubrandom_uniform/maxrandom_uniform/min*
T0*
_output_shapes
: 
t
random_uniform/mulMulrandom_uniform/RandomUniformrandom_uniform/sub*
T0*
_output_shapes

:	x
f
random_uniformAddrandom_uniform/mulrandom_uniform/min*
T0*
_output_shapes

:	x
|
Variable
VariableV2*
dtype0*
_output_shapes

:	x*
shared_name *
	container *
shape
:	x
¢
Variable/AssignAssignVariablerandom_uniform*
T0*
_output_shapes

:	x*
use_locking(*
validate_shape(*
_class
loc:@Variable
i
Variable/readIdentityVariable*
T0*
_output_shapes

:	x*
_class
loc:@Variable
g
random_uniform_1/shapeConst*
valueB"x   x   *
dtype0*
_output_shapes
:
Y
random_uniform_1/minConst*
valueB
 *č!¾*
dtype0*
_output_shapes
: 
Y
random_uniform_1/maxConst*
valueB
 *č!>*
dtype0*
_output_shapes
: 

random_uniform_1/RandomUniformRandomUniformrandom_uniform_1/shape*
T0*
dtype0*
_output_shapes

:xx*
seed2 *

seed 
h
random_uniform_1/subSubrandom_uniform_1/maxrandom_uniform_1/min*
T0*
_output_shapes
: 
z
random_uniform_1/mulMulrandom_uniform_1/RandomUniformrandom_uniform_1/sub*
T0*
_output_shapes

:xx
l
random_uniform_1Addrandom_uniform_1/mulrandom_uniform_1/min*
T0*
_output_shapes

:xx
~

Variable_1
VariableV2*
dtype0*
_output_shapes

:xx*
shared_name *
	container *
shape
:xx
Ŗ
Variable_1/AssignAssign
Variable_1random_uniform_1*
T0*
_output_shapes

:xx*
use_locking(*
validate_shape(*
_class
loc:@Variable_1
o
Variable_1/readIdentity
Variable_1*
T0*
_output_shapes

:xx*
_class
loc:@Variable_1
g
random_uniform_2/shapeConst*
valueB"x   x   *
dtype0*
_output_shapes
:
Y
random_uniform_2/minConst*
valueB
 *č!¾*
dtype0*
_output_shapes
: 
Y
random_uniform_2/maxConst*
valueB
 *č!>*
dtype0*
_output_shapes
: 

random_uniform_2/RandomUniformRandomUniformrandom_uniform_2/shape*
T0*
dtype0*
_output_shapes

:xx*
seed2 *

seed 
h
random_uniform_2/subSubrandom_uniform_2/maxrandom_uniform_2/min*
T0*
_output_shapes
: 
z
random_uniform_2/mulMulrandom_uniform_2/RandomUniformrandom_uniform_2/sub*
T0*
_output_shapes

:xx
l
random_uniform_2Addrandom_uniform_2/mulrandom_uniform_2/min*
T0*
_output_shapes

:xx
~

Variable_2
VariableV2*
dtype0*
_output_shapes

:xx*
shared_name *
	container *
shape
:xx
Ŗ
Variable_2/AssignAssign
Variable_2random_uniform_2*
T0*
_output_shapes

:xx*
use_locking(*
validate_shape(*
_class
loc:@Variable_2
o
Variable_2/readIdentity
Variable_2*
T0*
_output_shapes

:xx*
_class
loc:@Variable_2
g
random_uniform_3/shapeConst*
valueB"x      *
dtype0*
_output_shapes
:
Y
random_uniform_3/minConst*
valueB
 *ud¾*
dtype0*
_output_shapes
: 
Y
random_uniform_3/maxConst*
valueB
 *ud>*
dtype0*
_output_shapes
: 

random_uniform_3/RandomUniformRandomUniformrandom_uniform_3/shape*
T0*
dtype0*
_output_shapes

:x*
seed2 *

seed 
h
random_uniform_3/subSubrandom_uniform_3/maxrandom_uniform_3/min*
T0*
_output_shapes
: 
z
random_uniform_3/mulMulrandom_uniform_3/RandomUniformrandom_uniform_3/sub*
T0*
_output_shapes

:x
l
random_uniform_3Addrandom_uniform_3/mulrandom_uniform_3/min*
T0*
_output_shapes

:x
~

Variable_3
VariableV2*
dtype0*
_output_shapes

:x*
shared_name *
	container *
shape
:x
Ŗ
Variable_3/AssignAssign
Variable_3random_uniform_3*
T0*
_output_shapes

:x*
use_locking(*
validate_shape(*
_class
loc:@Variable_3
o
Variable_3/readIdentity
Variable_3*
T0*
_output_shapes

:x*
_class
loc:@Variable_3
`
random_uniform_4/shapeConst*
valueB:x*
dtype0*
_output_shapes
:
Y
random_uniform_4/minConst*
valueB
 *č!¾*
dtype0*
_output_shapes
: 
Y
random_uniform_4/maxConst*
valueB
 *č!>*
dtype0*
_output_shapes
: 

random_uniform_4/RandomUniformRandomUniformrandom_uniform_4/shape*
T0*
dtype0*
_output_shapes
:x*
seed2 *

seed 
h
random_uniform_4/subSubrandom_uniform_4/maxrandom_uniform_4/min*
T0*
_output_shapes
: 
v
random_uniform_4/mulMulrandom_uniform_4/RandomUniformrandom_uniform_4/sub*
T0*
_output_shapes
:x
h
random_uniform_4Addrandom_uniform_4/mulrandom_uniform_4/min*
T0*
_output_shapes
:x
v

Variable_4
VariableV2*
dtype0*
_output_shapes
:x*
shared_name *
	container *
shape:x
¦
Variable_4/AssignAssign
Variable_4random_uniform_4*
T0*
_output_shapes
:x*
use_locking(*
validate_shape(*
_class
loc:@Variable_4
k
Variable_4/readIdentity
Variable_4*
T0*
_output_shapes
:x*
_class
loc:@Variable_4
`
random_uniform_5/shapeConst*
valueB:x*
dtype0*
_output_shapes
:
Y
random_uniform_5/minConst*
valueB
 *č!¾*
dtype0*
_output_shapes
: 
Y
random_uniform_5/maxConst*
valueB
 *č!>*
dtype0*
_output_shapes
: 

random_uniform_5/RandomUniformRandomUniformrandom_uniform_5/shape*
T0*
dtype0*
_output_shapes
:x*
seed2 *

seed 
h
random_uniform_5/subSubrandom_uniform_5/maxrandom_uniform_5/min*
T0*
_output_shapes
: 
v
random_uniform_5/mulMulrandom_uniform_5/RandomUniformrandom_uniform_5/sub*
T0*
_output_shapes
:x
h
random_uniform_5Addrandom_uniform_5/mulrandom_uniform_5/min*
T0*
_output_shapes
:x
v

Variable_5
VariableV2*
dtype0*
_output_shapes
:x*
shared_name *
	container *
shape:x
¦
Variable_5/AssignAssign
Variable_5random_uniform_5*
T0*
_output_shapes
:x*
use_locking(*
validate_shape(*
_class
loc:@Variable_5
k
Variable_5/readIdentity
Variable_5*
T0*
_output_shapes
:x*
_class
loc:@Variable_5
`
random_uniform_6/shapeConst*
valueB:x*
dtype0*
_output_shapes
:
Y
random_uniform_6/minConst*
valueB
 *č!¾*
dtype0*
_output_shapes
: 
Y
random_uniform_6/maxConst*
valueB
 *č!>*
dtype0*
_output_shapes
: 

random_uniform_6/RandomUniformRandomUniformrandom_uniform_6/shape*
T0*
dtype0*
_output_shapes
:x*
seed2 *

seed 
h
random_uniform_6/subSubrandom_uniform_6/maxrandom_uniform_6/min*
T0*
_output_shapes
: 
v
random_uniform_6/mulMulrandom_uniform_6/RandomUniformrandom_uniform_6/sub*
T0*
_output_shapes
:x
h
random_uniform_6Addrandom_uniform_6/mulrandom_uniform_6/min*
T0*
_output_shapes
:x
v

Variable_6
VariableV2*
dtype0*
_output_shapes
:x*
shared_name *
	container *
shape:x
¦
Variable_6/AssignAssign
Variable_6random_uniform_6*
T0*
_output_shapes
:x*
use_locking(*
validate_shape(*
_class
loc:@Variable_6
k
Variable_6/readIdentity
Variable_6*
T0*
_output_shapes
:x*
_class
loc:@Variable_6
`
random_uniform_7/shapeConst*
valueB:*
dtype0*
_output_shapes
:
Y
random_uniform_7/minConst*
valueB
 *×³Żæ*
dtype0*
_output_shapes
: 
Y
random_uniform_7/maxConst*
valueB
 *×³Ż?*
dtype0*
_output_shapes
: 

random_uniform_7/RandomUniformRandomUniformrandom_uniform_7/shape*
T0*
dtype0*
_output_shapes
:*
seed2 *

seed 
h
random_uniform_7/subSubrandom_uniform_7/maxrandom_uniform_7/min*
T0*
_output_shapes
: 
v
random_uniform_7/mulMulrandom_uniform_7/RandomUniformrandom_uniform_7/sub*
T0*
_output_shapes
:
h
random_uniform_7Addrandom_uniform_7/mulrandom_uniform_7/min*
T0*
_output_shapes
:
v

Variable_7
VariableV2*
dtype0*
_output_shapes
:*
shared_name *
	container *
shape:
¦
Variable_7/AssignAssign
Variable_7random_uniform_7*
T0*
_output_shapes
:*
use_locking(*
validate_shape(*
_class
loc:@Variable_7
k
Variable_7/readIdentity
Variable_7*
T0*
_output_shapes
:*
_class
loc:@Variable_7
N
	keep_probPlaceholder*
dtype0*
_output_shapes
:*
shape:

MatMulMatMul	inputs_phVariable/read*
T0*
transpose_b( *
transpose_a( *'
_output_shapes
:’’’’’’’’’x
U
AddAddMatMulVariable_4/read*
T0*'
_output_shapes
:’’’’’’’’’x
C
ReluReluAdd*
T0*'
_output_shapes
:’’’’’’’’’x
Q
dropout/ShapeShapeRelu*
T0*
out_type0*
_output_shapes
:
_
dropout/random_uniform/minConst*
valueB
 *    *
dtype0*
_output_shapes
: 
_
dropout/random_uniform/maxConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 

$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape*
T0*
dtype0*'
_output_shapes
:’’’’’’’’’x*
seed2 *

seed 
z
dropout/random_uniform/subSubdropout/random_uniform/maxdropout/random_uniform/min*
T0*
_output_shapes
: 

dropout/random_uniform/mulMul$dropout/random_uniform/RandomUniformdropout/random_uniform/sub*
T0*'
_output_shapes
:’’’’’’’’’x

dropout/random_uniformAdddropout/random_uniform/muldropout/random_uniform/min*
T0*'
_output_shapes
:’’’’’’’’’x
X
dropout/addAdd	keep_probdropout/random_uniform*
T0*
_output_shapes
:
F
dropout/FloorFloordropout/add*
T0*
_output_shapes
:
J
dropout/divRealDivRelu	keep_prob*
T0*
_output_shapes
:
`
dropout/mulMuldropout/divdropout/Floor*
T0*'
_output_shapes
:’’’’’’’’’x

MatMul_1MatMuldropout/mulVariable_1/read*
T0*
transpose_b( *
transpose_a( *'
_output_shapes
:’’’’’’’’’x
Y
Add_1AddMatMul_1Variable_5/read*
T0*'
_output_shapes
:’’’’’’’’’x
G
Relu_1ReluAdd_1*
T0*'
_output_shapes
:’’’’’’’’’x
U
dropout_1/ShapeShapeRelu_1*
T0*
out_type0*
_output_shapes
:
a
dropout_1/random_uniform/minConst*
valueB
 *    *
dtype0*
_output_shapes
: 
a
dropout_1/random_uniform/maxConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
 
&dropout_1/random_uniform/RandomUniformRandomUniformdropout_1/Shape*
T0*
dtype0*'
_output_shapes
:’’’’’’’’’x*
seed2 *

seed 

dropout_1/random_uniform/subSubdropout_1/random_uniform/maxdropout_1/random_uniform/min*
T0*
_output_shapes
: 

dropout_1/random_uniform/mulMul&dropout_1/random_uniform/RandomUniformdropout_1/random_uniform/sub*
T0*'
_output_shapes
:’’’’’’’’’x

dropout_1/random_uniformAdddropout_1/random_uniform/muldropout_1/random_uniform/min*
T0*'
_output_shapes
:’’’’’’’’’x
\
dropout_1/addAdd	keep_probdropout_1/random_uniform*
T0*
_output_shapes
:
J
dropout_1/FloorFloordropout_1/add*
T0*
_output_shapes
:
N
dropout_1/divRealDivRelu_1	keep_prob*
T0*
_output_shapes
:
f
dropout_1/mulMuldropout_1/divdropout_1/Floor*
T0*'
_output_shapes
:’’’’’’’’’x

MatMul_2MatMuldropout_1/mulVariable_1/read*
T0*
transpose_b( *
transpose_a( *'
_output_shapes
:’’’’’’’’’x
Y
Add_2AddMatMul_2Variable_5/read*
T0*'
_output_shapes
:’’’’’’’’’x
G
Relu_2ReluAdd_2*
T0*'
_output_shapes
:’’’’’’’’’x
U
dropout_2/ShapeShapeRelu_2*
T0*
out_type0*
_output_shapes
:
a
dropout_2/random_uniform/minConst*
valueB
 *    *
dtype0*
_output_shapes
: 
a
dropout_2/random_uniform/maxConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
 
&dropout_2/random_uniform/RandomUniformRandomUniformdropout_2/Shape*
T0*
dtype0*'
_output_shapes
:’’’’’’’’’x*
seed2 *

seed 

dropout_2/random_uniform/subSubdropout_2/random_uniform/maxdropout_2/random_uniform/min*
T0*
_output_shapes
: 

dropout_2/random_uniform/mulMul&dropout_2/random_uniform/RandomUniformdropout_2/random_uniform/sub*
T0*'
_output_shapes
:’’’’’’’’’x

dropout_2/random_uniformAdddropout_2/random_uniform/muldropout_2/random_uniform/min*
T0*'
_output_shapes
:’’’’’’’’’x
\
dropout_2/addAdd	keep_probdropout_2/random_uniform*
T0*
_output_shapes
:
J
dropout_2/FloorFloordropout_2/add*
T0*
_output_shapes
:
N
dropout_2/divRealDivRelu_2	keep_prob*
T0*
_output_shapes
:
f
dropout_2/mulMuldropout_2/divdropout_2/Floor*
T0*'
_output_shapes
:’’’’’’’’’x

MatMul_3MatMuldropout_2/mulVariable_3/read*
T0*
transpose_b( *
transpose_a( *'
_output_shapes
:’’’’’’’’’
Y
add_3AddMatMul_3Variable_7/read*
T0*'
_output_shapes
:’’’’’’’’’
^
logistic_loss/zeros_like	ZerosLikeadd_3*
T0*'
_output_shapes
:’’’’’’’’’
}
logistic_loss/GreaterEqualGreaterEqualadd_3logistic_loss/zeros_like*
T0*'
_output_shapes
:’’’’’’’’’

logistic_loss/SelectSelectlogistic_loss/GreaterEqualadd_3logistic_loss/zeros_like*
T0*'
_output_shapes
:’’’’’’’’’
Q
logistic_loss/NegNegadd_3*
T0*'
_output_shapes
:’’’’’’’’’

logistic_loss/Select_1Selectlogistic_loss/GreaterEquallogistic_loss/Negadd_3*
T0*'
_output_shapes
:’’’’’’’’’
\
logistic_loss/mulMuladd_3	labels_ph*
T0*'
_output_shapes
:’’’’’’’’’
s
logistic_loss/subSublogistic_loss/Selectlogistic_loss/mul*
T0*'
_output_shapes
:’’’’’’’’’
b
logistic_loss/ExpExplogistic_loss/Select_1*
T0*'
_output_shapes
:’’’’’’’’’
a
logistic_loss/Log1pLog1plogistic_loss/Exp*
T0*'
_output_shapes
:’’’’’’’’’
n
logistic_lossAddlogistic_loss/sublogistic_loss/Log1p*
T0*'
_output_shapes
:’’’’’’’’’
V
ConstConst*
valueB"       *
dtype0*
_output_shapes
:
`
MeanMeanlogistic_lossConst*
	keep_dims( *
T0*
_output_shapes
: *

Tidx0
R
gradients/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
X
gradients/grad_ys_0Const*
valueB
 *  ?*
dtype0*
_output_shapes
: 
o
gradients/FillFillgradients/Shapegradients/grad_ys_0*
T0*
_output_shapes
: *

index_type0
r
!gradients/Mean_grad/Reshape/shapeConst*
valueB"      *
dtype0*
_output_shapes
:

gradients/Mean_grad/ReshapeReshapegradients/Fill!gradients/Mean_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes

:
f
gradients/Mean_grad/ShapeShapelogistic_loss*
T0*
out_type0*
_output_shapes
:

gradients/Mean_grad/TileTilegradients/Mean_grad/Reshapegradients/Mean_grad/Shape*
T0*'
_output_shapes
:’’’’’’’’’*

Tmultiples0
h
gradients/Mean_grad/Shape_1Shapelogistic_loss*
T0*
out_type0*
_output_shapes
:
^
gradients/Mean_grad/Shape_2Const*
valueB *
dtype0*
_output_shapes
: 
c
gradients/Mean_grad/ConstConst*
valueB: *
dtype0*
_output_shapes
:

gradients/Mean_grad/ProdProdgradients/Mean_grad/Shape_1gradients/Mean_grad/Const*
	keep_dims( *
T0*
_output_shapes
: *

Tidx0
e
gradients/Mean_grad/Const_1Const*
valueB: *
dtype0*
_output_shapes
:

gradients/Mean_grad/Prod_1Prodgradients/Mean_grad/Shape_2gradients/Mean_grad/Const_1*
	keep_dims( *
T0*
_output_shapes
: *

Tidx0
_
gradients/Mean_grad/Maximum/yConst*
value	B :*
dtype0*
_output_shapes
: 

gradients/Mean_grad/MaximumMaximumgradients/Mean_grad/Prod_1gradients/Mean_grad/Maximum/y*
T0*
_output_shapes
: 

gradients/Mean_grad/floordivFloorDivgradients/Mean_grad/Prodgradients/Mean_grad/Maximum*
T0*
_output_shapes
: 
~
gradients/Mean_grad/CastCastgradients/Mean_grad/floordiv*

DstT0*
_output_shapes
: *
Truncate( *

SrcT0

gradients/Mean_grad/truedivRealDivgradients/Mean_grad/Tilegradients/Mean_grad/Cast*
T0*'
_output_shapes
:’’’’’’’’’
s
"gradients/logistic_loss_grad/ShapeShapelogistic_loss/sub*
T0*
out_type0*
_output_shapes
:
w
$gradients/logistic_loss_grad/Shape_1Shapelogistic_loss/Log1p*
T0*
out_type0*
_output_shapes
:
Ņ
2gradients/logistic_loss_grad/BroadcastGradientArgsBroadcastGradientArgs"gradients/logistic_loss_grad/Shape$gradients/logistic_loss_grad/Shape_1*
T0*2
_output_shapes 
:’’’’’’’’’:’’’’’’’’’
ø
 gradients/logistic_loss_grad/SumSumgradients/Mean_grad/truediv2gradients/logistic_loss_grad/BroadcastGradientArgs*
	keep_dims( *
T0*
_output_shapes
:*

Tidx0
µ
$gradients/logistic_loss_grad/ReshapeReshape gradients/logistic_loss_grad/Sum"gradients/logistic_loss_grad/Shape*
T0*
Tshape0*'
_output_shapes
:’’’’’’’’’
¼
"gradients/logistic_loss_grad/Sum_1Sumgradients/Mean_grad/truediv4gradients/logistic_loss_grad/BroadcastGradientArgs:1*
	keep_dims( *
T0*
_output_shapes
:*

Tidx0
»
&gradients/logistic_loss_grad/Reshape_1Reshape"gradients/logistic_loss_grad/Sum_1$gradients/logistic_loss_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:’’’’’’’’’

-gradients/logistic_loss_grad/tuple/group_depsNoOp%^gradients/logistic_loss_grad/Reshape'^gradients/logistic_loss_grad/Reshape_1

5gradients/logistic_loss_grad/tuple/control_dependencyIdentity$gradients/logistic_loss_grad/Reshape.^gradients/logistic_loss_grad/tuple/group_deps*
T0*'
_output_shapes
:’’’’’’’’’*7
_class-
+)loc:@gradients/logistic_loss_grad/Reshape

7gradients/logistic_loss_grad/tuple/control_dependency_1Identity&gradients/logistic_loss_grad/Reshape_1.^gradients/logistic_loss_grad/tuple/group_deps*
T0*'
_output_shapes
:’’’’’’’’’*9
_class/
-+loc:@gradients/logistic_loss_grad/Reshape_1
z
&gradients/logistic_loss/sub_grad/ShapeShapelogistic_loss/Select*
T0*
out_type0*
_output_shapes
:
y
(gradients/logistic_loss/sub_grad/Shape_1Shapelogistic_loss/mul*
T0*
out_type0*
_output_shapes
:
Ž
6gradients/logistic_loss/sub_grad/BroadcastGradientArgsBroadcastGradientArgs&gradients/logistic_loss/sub_grad/Shape(gradients/logistic_loss/sub_grad/Shape_1*
T0*2
_output_shapes 
:’’’’’’’’’:’’’’’’’’’
Ś
$gradients/logistic_loss/sub_grad/SumSum5gradients/logistic_loss_grad/tuple/control_dependency6gradients/logistic_loss/sub_grad/BroadcastGradientArgs*
	keep_dims( *
T0*
_output_shapes
:*

Tidx0
Į
(gradients/logistic_loss/sub_grad/ReshapeReshape$gradients/logistic_loss/sub_grad/Sum&gradients/logistic_loss/sub_grad/Shape*
T0*
Tshape0*'
_output_shapes
:’’’’’’’’’
Ž
&gradients/logistic_loss/sub_grad/Sum_1Sum5gradients/logistic_loss_grad/tuple/control_dependency8gradients/logistic_loss/sub_grad/BroadcastGradientArgs:1*
	keep_dims( *
T0*
_output_shapes
:*

Tidx0
v
$gradients/logistic_loss/sub_grad/NegNeg&gradients/logistic_loss/sub_grad/Sum_1*
T0*
_output_shapes
:
Å
*gradients/logistic_loss/sub_grad/Reshape_1Reshape$gradients/logistic_loss/sub_grad/Neg(gradients/logistic_loss/sub_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:’’’’’’’’’

1gradients/logistic_loss/sub_grad/tuple/group_depsNoOp)^gradients/logistic_loss/sub_grad/Reshape+^gradients/logistic_loss/sub_grad/Reshape_1

9gradients/logistic_loss/sub_grad/tuple/control_dependencyIdentity(gradients/logistic_loss/sub_grad/Reshape2^gradients/logistic_loss/sub_grad/tuple/group_deps*
T0*'
_output_shapes
:’’’’’’’’’*;
_class1
/-loc:@gradients/logistic_loss/sub_grad/Reshape

;gradients/logistic_loss/sub_grad/tuple/control_dependency_1Identity*gradients/logistic_loss/sub_grad/Reshape_12^gradients/logistic_loss/sub_grad/tuple/group_deps*
T0*'
_output_shapes
:’’’’’’’’’*=
_class3
1/loc:@gradients/logistic_loss/sub_grad/Reshape_1
§
(gradients/logistic_loss/Log1p_grad/add/xConst8^gradients/logistic_loss_grad/tuple/control_dependency_1*
valueB
 *  ?*
dtype0*
_output_shapes
: 

&gradients/logistic_loss/Log1p_grad/addAdd(gradients/logistic_loss/Log1p_grad/add/xlogistic_loss/Exp*
T0*'
_output_shapes
:’’’’’’’’’

-gradients/logistic_loss/Log1p_grad/Reciprocal
Reciprocal&gradients/logistic_loss/Log1p_grad/add*
T0*'
_output_shapes
:’’’’’’’’’
Ē
&gradients/logistic_loss/Log1p_grad/mulMul7gradients/logistic_loss_grad/tuple/control_dependency_1-gradients/logistic_loss/Log1p_grad/Reciprocal*
T0*'
_output_shapes
:’’’’’’’’’
t
.gradients/logistic_loss/Select_grad/zeros_like	ZerosLikeadd_3*
T0*'
_output_shapes
:’’’’’’’’’
ķ
*gradients/logistic_loss/Select_grad/SelectSelectlogistic_loss/GreaterEqual9gradients/logistic_loss/sub_grad/tuple/control_dependency.gradients/logistic_loss/Select_grad/zeros_like*
T0*'
_output_shapes
:’’’’’’’’’
ļ
,gradients/logistic_loss/Select_grad/Select_1Selectlogistic_loss/GreaterEqual.gradients/logistic_loss/Select_grad/zeros_like9gradients/logistic_loss/sub_grad/tuple/control_dependency*
T0*'
_output_shapes
:’’’’’’’’’

4gradients/logistic_loss/Select_grad/tuple/group_depsNoOp+^gradients/logistic_loss/Select_grad/Select-^gradients/logistic_loss/Select_grad/Select_1

<gradients/logistic_loss/Select_grad/tuple/control_dependencyIdentity*gradients/logistic_loss/Select_grad/Select5^gradients/logistic_loss/Select_grad/tuple/group_deps*
T0*'
_output_shapes
:’’’’’’’’’*=
_class3
1/loc:@gradients/logistic_loss/Select_grad/Select
¢
>gradients/logistic_loss/Select_grad/tuple/control_dependency_1Identity,gradients/logistic_loss/Select_grad/Select_15^gradients/logistic_loss/Select_grad/tuple/group_deps*
T0*'
_output_shapes
:’’’’’’’’’*?
_class5
31loc:@gradients/logistic_loss/Select_grad/Select_1
k
&gradients/logistic_loss/mul_grad/ShapeShapeadd_3*
T0*
out_type0*
_output_shapes
:
q
(gradients/logistic_loss/mul_grad/Shape_1Shape	labels_ph*
T0*
out_type0*
_output_shapes
:
Ž
6gradients/logistic_loss/mul_grad/BroadcastGradientArgsBroadcastGradientArgs&gradients/logistic_loss/mul_grad/Shape(gradients/logistic_loss/mul_grad/Shape_1*
T0*2
_output_shapes 
:’’’’’’’’’:’’’’’’’’’
„
$gradients/logistic_loss/mul_grad/MulMul;gradients/logistic_loss/sub_grad/tuple/control_dependency_1	labels_ph*
T0*'
_output_shapes
:’’’’’’’’’
É
$gradients/logistic_loss/mul_grad/SumSum$gradients/logistic_loss/mul_grad/Mul6gradients/logistic_loss/mul_grad/BroadcastGradientArgs*
	keep_dims( *
T0*
_output_shapes
:*

Tidx0
Į
(gradients/logistic_loss/mul_grad/ReshapeReshape$gradients/logistic_loss/mul_grad/Sum&gradients/logistic_loss/mul_grad/Shape*
T0*
Tshape0*'
_output_shapes
:’’’’’’’’’
£
&gradients/logistic_loss/mul_grad/Mul_1Muladd_3;gradients/logistic_loss/sub_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:’’’’’’’’’
Ļ
&gradients/logistic_loss/mul_grad/Sum_1Sum&gradients/logistic_loss/mul_grad/Mul_18gradients/logistic_loss/mul_grad/BroadcastGradientArgs:1*
	keep_dims( *
T0*
_output_shapes
:*

Tidx0
Ē
*gradients/logistic_loss/mul_grad/Reshape_1Reshape&gradients/logistic_loss/mul_grad/Sum_1(gradients/logistic_loss/mul_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:’’’’’’’’’

1gradients/logistic_loss/mul_grad/tuple/group_depsNoOp)^gradients/logistic_loss/mul_grad/Reshape+^gradients/logistic_loss/mul_grad/Reshape_1

9gradients/logistic_loss/mul_grad/tuple/control_dependencyIdentity(gradients/logistic_loss/mul_grad/Reshape2^gradients/logistic_loss/mul_grad/tuple/group_deps*
T0*'
_output_shapes
:’’’’’’’’’*;
_class1
/-loc:@gradients/logistic_loss/mul_grad/Reshape

;gradients/logistic_loss/mul_grad/tuple/control_dependency_1Identity*gradients/logistic_loss/mul_grad/Reshape_12^gradients/logistic_loss/mul_grad/tuple/group_deps*
T0*'
_output_shapes
:’’’’’’’’’*=
_class3
1/loc:@gradients/logistic_loss/mul_grad/Reshape_1

$gradients/logistic_loss/Exp_grad/mulMul&gradients/logistic_loss/Log1p_grad/mullogistic_loss/Exp*
T0*'
_output_shapes
:’’’’’’’’’

0gradients/logistic_loss/Select_1_grad/zeros_like	ZerosLikelogistic_loss/Neg*
T0*'
_output_shapes
:’’’’’’’’’
Ü
,gradients/logistic_loss/Select_1_grad/SelectSelectlogistic_loss/GreaterEqual$gradients/logistic_loss/Exp_grad/mul0gradients/logistic_loss/Select_1_grad/zeros_like*
T0*'
_output_shapes
:’’’’’’’’’
Ž
.gradients/logistic_loss/Select_1_grad/Select_1Selectlogistic_loss/GreaterEqual0gradients/logistic_loss/Select_1_grad/zeros_like$gradients/logistic_loss/Exp_grad/mul*
T0*'
_output_shapes
:’’’’’’’’’

6gradients/logistic_loss/Select_1_grad/tuple/group_depsNoOp-^gradients/logistic_loss/Select_1_grad/Select/^gradients/logistic_loss/Select_1_grad/Select_1
¤
>gradients/logistic_loss/Select_1_grad/tuple/control_dependencyIdentity,gradients/logistic_loss/Select_1_grad/Select7^gradients/logistic_loss/Select_1_grad/tuple/group_deps*
T0*'
_output_shapes
:’’’’’’’’’*?
_class5
31loc:@gradients/logistic_loss/Select_1_grad/Select
Ŗ
@gradients/logistic_loss/Select_1_grad/tuple/control_dependency_1Identity.gradients/logistic_loss/Select_1_grad/Select_17^gradients/logistic_loss/Select_1_grad/tuple/group_deps*
T0*'
_output_shapes
:’’’’’’’’’*A
_class7
53loc:@gradients/logistic_loss/Select_1_grad/Select_1

$gradients/logistic_loss/Neg_grad/NegNeg>gradients/logistic_loss/Select_1_grad/tuple/control_dependency*
T0*'
_output_shapes
:’’’’’’’’’
ń
gradients/AddNAddN<gradients/logistic_loss/Select_grad/tuple/control_dependency9gradients/logistic_loss/mul_grad/tuple/control_dependency@gradients/logistic_loss/Select_1_grad/tuple/control_dependency_1$gradients/logistic_loss/Neg_grad/Neg*
T0*
N*'
_output_shapes
:’’’’’’’’’*=
_class3
1/loc:@gradients/logistic_loss/Select_grad/Select
b
gradients/add_3_grad/ShapeShapeMatMul_3*
T0*
out_type0*
_output_shapes
:
f
gradients/add_3_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
ŗ
*gradients/add_3_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_3_grad/Shapegradients/add_3_grad/Shape_1*
T0*2
_output_shapes 
:’’’’’’’’’:’’’’’’’’’

gradients/add_3_grad/SumSumgradients/AddN*gradients/add_3_grad/BroadcastGradientArgs*
	keep_dims( *
T0*
_output_shapes
:*

Tidx0

gradients/add_3_grad/ReshapeReshapegradients/add_3_grad/Sumgradients/add_3_grad/Shape*
T0*
Tshape0*'
_output_shapes
:’’’’’’’’’

gradients/add_3_grad/Sum_1Sumgradients/AddN,gradients/add_3_grad/BroadcastGradientArgs:1*
	keep_dims( *
T0*
_output_shapes
:*

Tidx0

gradients/add_3_grad/Reshape_1Reshapegradients/add_3_grad/Sum_1gradients/add_3_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:
m
%gradients/add_3_grad/tuple/group_depsNoOp^gradients/add_3_grad/Reshape^gradients/add_3_grad/Reshape_1
ā
-gradients/add_3_grad/tuple/control_dependencyIdentitygradients/add_3_grad/Reshape&^gradients/add_3_grad/tuple/group_deps*
T0*'
_output_shapes
:’’’’’’’’’*/
_class%
#!loc:@gradients/add_3_grad/Reshape
Ū
/gradients/add_3_grad/tuple/control_dependency_1Identitygradients/add_3_grad/Reshape_1&^gradients/add_3_grad/tuple/group_deps*
T0*
_output_shapes
:*1
_class'
%#loc:@gradients/add_3_grad/Reshape_1
Ą
gradients/MatMul_3_grad/MatMulMatMul-gradients/add_3_grad/tuple/control_dependencyVariable_3/read*
T0*
transpose_b(*
transpose_a( *'
_output_shapes
:’’’’’’’’’x
·
 gradients/MatMul_3_grad/MatMul_1MatMuldropout_2/mul-gradients/add_3_grad/tuple/control_dependency*
T0*
transpose_b( *
transpose_a(*
_output_shapes

:x
t
(gradients/MatMul_3_grad/tuple/group_depsNoOp^gradients/MatMul_3_grad/MatMul!^gradients/MatMul_3_grad/MatMul_1
ģ
0gradients/MatMul_3_grad/tuple/control_dependencyIdentitygradients/MatMul_3_grad/MatMul)^gradients/MatMul_3_grad/tuple/group_deps*
T0*'
_output_shapes
:’’’’’’’’’x*1
_class'
%#loc:@gradients/MatMul_3_grad/MatMul
é
2gradients/MatMul_3_grad/tuple/control_dependency_1Identity gradients/MatMul_3_grad/MatMul_1)^gradients/MatMul_3_grad/tuple/group_deps*
T0*
_output_shapes

:x*3
_class)
'%loc:@gradients/MatMul_3_grad/MatMul_1
x
"gradients/dropout_2/mul_grad/ShapeShapedropout_2/div*
T0*
out_type0*#
_output_shapes
:’’’’’’’’’
|
$gradients/dropout_2/mul_grad/Shape_1Shapedropout_2/Floor*
T0*
out_type0*#
_output_shapes
:’’’’’’’’’
Ņ
2gradients/dropout_2/mul_grad/BroadcastGradientArgsBroadcastGradientArgs"gradients/dropout_2/mul_grad/Shape$gradients/dropout_2/mul_grad/Shape_1*
T0*2
_output_shapes 
:’’’’’’’’’:’’’’’’’’’

 gradients/dropout_2/mul_grad/MulMul0gradients/MatMul_3_grad/tuple/control_dependencydropout_2/Floor*
T0*
_output_shapes
:
½
 gradients/dropout_2/mul_grad/SumSum gradients/dropout_2/mul_grad/Mul2gradients/dropout_2/mul_grad/BroadcastGradientArgs*
	keep_dims( *
T0*
_output_shapes
:*

Tidx0
¦
$gradients/dropout_2/mul_grad/ReshapeReshape gradients/dropout_2/mul_grad/Sum"gradients/dropout_2/mul_grad/Shape*
T0*
Tshape0*
_output_shapes
:

"gradients/dropout_2/mul_grad/Mul_1Muldropout_2/div0gradients/MatMul_3_grad/tuple/control_dependency*
T0*
_output_shapes
:
Ć
"gradients/dropout_2/mul_grad/Sum_1Sum"gradients/dropout_2/mul_grad/Mul_14gradients/dropout_2/mul_grad/BroadcastGradientArgs:1*
	keep_dims( *
T0*
_output_shapes
:*

Tidx0
¬
&gradients/dropout_2/mul_grad/Reshape_1Reshape"gradients/dropout_2/mul_grad/Sum_1$gradients/dropout_2/mul_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:

-gradients/dropout_2/mul_grad/tuple/group_depsNoOp%^gradients/dropout_2/mul_grad/Reshape'^gradients/dropout_2/mul_grad/Reshape_1
ó
5gradients/dropout_2/mul_grad/tuple/control_dependencyIdentity$gradients/dropout_2/mul_grad/Reshape.^gradients/dropout_2/mul_grad/tuple/group_deps*
T0*
_output_shapes
:*7
_class-
+)loc:@gradients/dropout_2/mul_grad/Reshape
ł
7gradients/dropout_2/mul_grad/tuple/control_dependency_1Identity&gradients/dropout_2/mul_grad/Reshape_1.^gradients/dropout_2/mul_grad/tuple/group_deps*
T0*
_output_shapes
:*9
_class/
-+loc:@gradients/dropout_2/mul_grad/Reshape_1
h
"gradients/dropout_2/div_grad/ShapeShapeRelu_2*
T0*
out_type0*
_output_shapes
:
v
$gradients/dropout_2/div_grad/Shape_1Shape	keep_prob*
T0*
out_type0*#
_output_shapes
:’’’’’’’’’
Ņ
2gradients/dropout_2/div_grad/BroadcastGradientArgsBroadcastGradientArgs"gradients/dropout_2/div_grad/Shape$gradients/dropout_2/div_grad/Shape_1*
T0*2
_output_shapes 
:’’’’’’’’’:’’’’’’’’’

$gradients/dropout_2/div_grad/RealDivRealDiv5gradients/dropout_2/mul_grad/tuple/control_dependency	keep_prob*
T0*
_output_shapes
:
Į
 gradients/dropout_2/div_grad/SumSum$gradients/dropout_2/div_grad/RealDiv2gradients/dropout_2/div_grad/BroadcastGradientArgs*
	keep_dims( *
T0*
_output_shapes
:*

Tidx0
µ
$gradients/dropout_2/div_grad/ReshapeReshape gradients/dropout_2/div_grad/Sum"gradients/dropout_2/div_grad/Shape*
T0*
Tshape0*'
_output_shapes
:’’’’’’’’’x
a
 gradients/dropout_2/div_grad/NegNegRelu_2*
T0*'
_output_shapes
:’’’’’’’’’x

&gradients/dropout_2/div_grad/RealDiv_1RealDiv gradients/dropout_2/div_grad/Neg	keep_prob*
T0*
_output_shapes
:

&gradients/dropout_2/div_grad/RealDiv_2RealDiv&gradients/dropout_2/div_grad/RealDiv_1	keep_prob*
T0*
_output_shapes
:
©
 gradients/dropout_2/div_grad/mulMul5gradients/dropout_2/mul_grad/tuple/control_dependency&gradients/dropout_2/div_grad/RealDiv_2*
T0*
_output_shapes
:
Į
"gradients/dropout_2/div_grad/Sum_1Sum gradients/dropout_2/div_grad/mul4gradients/dropout_2/div_grad/BroadcastGradientArgs:1*
	keep_dims( *
T0*
_output_shapes
:*

Tidx0
¬
&gradients/dropout_2/div_grad/Reshape_1Reshape"gradients/dropout_2/div_grad/Sum_1$gradients/dropout_2/div_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:

-gradients/dropout_2/div_grad/tuple/group_depsNoOp%^gradients/dropout_2/div_grad/Reshape'^gradients/dropout_2/div_grad/Reshape_1

5gradients/dropout_2/div_grad/tuple/control_dependencyIdentity$gradients/dropout_2/div_grad/Reshape.^gradients/dropout_2/div_grad/tuple/group_deps*
T0*'
_output_shapes
:’’’’’’’’’x*7
_class-
+)loc:@gradients/dropout_2/div_grad/Reshape
ł
7gradients/dropout_2/div_grad/tuple/control_dependency_1Identity&gradients/dropout_2/div_grad/Reshape_1.^gradients/dropout_2/div_grad/tuple/group_deps*
T0*
_output_shapes
:*9
_class/
-+loc:@gradients/dropout_2/div_grad/Reshape_1

gradients/Relu_2_grad/ReluGradReluGrad5gradients/dropout_2/div_grad/tuple/control_dependencyRelu_2*
T0*'
_output_shapes
:’’’’’’’’’x
b
gradients/Add_2_grad/ShapeShapeMatMul_2*
T0*
out_type0*
_output_shapes
:
f
gradients/Add_2_grad/Shape_1Const*
valueB:x*
dtype0*
_output_shapes
:
ŗ
*gradients/Add_2_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/Add_2_grad/Shapegradients/Add_2_grad/Shape_1*
T0*2
_output_shapes 
:’’’’’’’’’:’’’’’’’’’
«
gradients/Add_2_grad/SumSumgradients/Relu_2_grad/ReluGrad*gradients/Add_2_grad/BroadcastGradientArgs*
	keep_dims( *
T0*
_output_shapes
:*

Tidx0

gradients/Add_2_grad/ReshapeReshapegradients/Add_2_grad/Sumgradients/Add_2_grad/Shape*
T0*
Tshape0*'
_output_shapes
:’’’’’’’’’x
Æ
gradients/Add_2_grad/Sum_1Sumgradients/Relu_2_grad/ReluGrad,gradients/Add_2_grad/BroadcastGradientArgs:1*
	keep_dims( *
T0*
_output_shapes
:*

Tidx0

gradients/Add_2_grad/Reshape_1Reshapegradients/Add_2_grad/Sum_1gradients/Add_2_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:x
m
%gradients/Add_2_grad/tuple/group_depsNoOp^gradients/Add_2_grad/Reshape^gradients/Add_2_grad/Reshape_1
ā
-gradients/Add_2_grad/tuple/control_dependencyIdentitygradients/Add_2_grad/Reshape&^gradients/Add_2_grad/tuple/group_deps*
T0*'
_output_shapes
:’’’’’’’’’x*/
_class%
#!loc:@gradients/Add_2_grad/Reshape
Ū
/gradients/Add_2_grad/tuple/control_dependency_1Identitygradients/Add_2_grad/Reshape_1&^gradients/Add_2_grad/tuple/group_deps*
T0*
_output_shapes
:x*1
_class'
%#loc:@gradients/Add_2_grad/Reshape_1
Ą
gradients/MatMul_2_grad/MatMulMatMul-gradients/Add_2_grad/tuple/control_dependencyVariable_1/read*
T0*
transpose_b(*
transpose_a( *'
_output_shapes
:’’’’’’’’’x
·
 gradients/MatMul_2_grad/MatMul_1MatMuldropout_1/mul-gradients/Add_2_grad/tuple/control_dependency*
T0*
transpose_b( *
transpose_a(*
_output_shapes

:xx
t
(gradients/MatMul_2_grad/tuple/group_depsNoOp^gradients/MatMul_2_grad/MatMul!^gradients/MatMul_2_grad/MatMul_1
ģ
0gradients/MatMul_2_grad/tuple/control_dependencyIdentitygradients/MatMul_2_grad/MatMul)^gradients/MatMul_2_grad/tuple/group_deps*
T0*'
_output_shapes
:’’’’’’’’’x*1
_class'
%#loc:@gradients/MatMul_2_grad/MatMul
é
2gradients/MatMul_2_grad/tuple/control_dependency_1Identity gradients/MatMul_2_grad/MatMul_1)^gradients/MatMul_2_grad/tuple/group_deps*
T0*
_output_shapes

:xx*3
_class)
'%loc:@gradients/MatMul_2_grad/MatMul_1
x
"gradients/dropout_1/mul_grad/ShapeShapedropout_1/div*
T0*
out_type0*#
_output_shapes
:’’’’’’’’’
|
$gradients/dropout_1/mul_grad/Shape_1Shapedropout_1/Floor*
T0*
out_type0*#
_output_shapes
:’’’’’’’’’
Ņ
2gradients/dropout_1/mul_grad/BroadcastGradientArgsBroadcastGradientArgs"gradients/dropout_1/mul_grad/Shape$gradients/dropout_1/mul_grad/Shape_1*
T0*2
_output_shapes 
:’’’’’’’’’:’’’’’’’’’

 gradients/dropout_1/mul_grad/MulMul0gradients/MatMul_2_grad/tuple/control_dependencydropout_1/Floor*
T0*
_output_shapes
:
½
 gradients/dropout_1/mul_grad/SumSum gradients/dropout_1/mul_grad/Mul2gradients/dropout_1/mul_grad/BroadcastGradientArgs*
	keep_dims( *
T0*
_output_shapes
:*

Tidx0
¦
$gradients/dropout_1/mul_grad/ReshapeReshape gradients/dropout_1/mul_grad/Sum"gradients/dropout_1/mul_grad/Shape*
T0*
Tshape0*
_output_shapes
:

"gradients/dropout_1/mul_grad/Mul_1Muldropout_1/div0gradients/MatMul_2_grad/tuple/control_dependency*
T0*
_output_shapes
:
Ć
"gradients/dropout_1/mul_grad/Sum_1Sum"gradients/dropout_1/mul_grad/Mul_14gradients/dropout_1/mul_grad/BroadcastGradientArgs:1*
	keep_dims( *
T0*
_output_shapes
:*

Tidx0
¬
&gradients/dropout_1/mul_grad/Reshape_1Reshape"gradients/dropout_1/mul_grad/Sum_1$gradients/dropout_1/mul_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:

-gradients/dropout_1/mul_grad/tuple/group_depsNoOp%^gradients/dropout_1/mul_grad/Reshape'^gradients/dropout_1/mul_grad/Reshape_1
ó
5gradients/dropout_1/mul_grad/tuple/control_dependencyIdentity$gradients/dropout_1/mul_grad/Reshape.^gradients/dropout_1/mul_grad/tuple/group_deps*
T0*
_output_shapes
:*7
_class-
+)loc:@gradients/dropout_1/mul_grad/Reshape
ł
7gradients/dropout_1/mul_grad/tuple/control_dependency_1Identity&gradients/dropout_1/mul_grad/Reshape_1.^gradients/dropout_1/mul_grad/tuple/group_deps*
T0*
_output_shapes
:*9
_class/
-+loc:@gradients/dropout_1/mul_grad/Reshape_1
h
"gradients/dropout_1/div_grad/ShapeShapeRelu_1*
T0*
out_type0*
_output_shapes
:
v
$gradients/dropout_1/div_grad/Shape_1Shape	keep_prob*
T0*
out_type0*#
_output_shapes
:’’’’’’’’’
Ņ
2gradients/dropout_1/div_grad/BroadcastGradientArgsBroadcastGradientArgs"gradients/dropout_1/div_grad/Shape$gradients/dropout_1/div_grad/Shape_1*
T0*2
_output_shapes 
:’’’’’’’’’:’’’’’’’’’

$gradients/dropout_1/div_grad/RealDivRealDiv5gradients/dropout_1/mul_grad/tuple/control_dependency	keep_prob*
T0*
_output_shapes
:
Į
 gradients/dropout_1/div_grad/SumSum$gradients/dropout_1/div_grad/RealDiv2gradients/dropout_1/div_grad/BroadcastGradientArgs*
	keep_dims( *
T0*
_output_shapes
:*

Tidx0
µ
$gradients/dropout_1/div_grad/ReshapeReshape gradients/dropout_1/div_grad/Sum"gradients/dropout_1/div_grad/Shape*
T0*
Tshape0*'
_output_shapes
:’’’’’’’’’x
a
 gradients/dropout_1/div_grad/NegNegRelu_1*
T0*'
_output_shapes
:’’’’’’’’’x

&gradients/dropout_1/div_grad/RealDiv_1RealDiv gradients/dropout_1/div_grad/Neg	keep_prob*
T0*
_output_shapes
:

&gradients/dropout_1/div_grad/RealDiv_2RealDiv&gradients/dropout_1/div_grad/RealDiv_1	keep_prob*
T0*
_output_shapes
:
©
 gradients/dropout_1/div_grad/mulMul5gradients/dropout_1/mul_grad/tuple/control_dependency&gradients/dropout_1/div_grad/RealDiv_2*
T0*
_output_shapes
:
Į
"gradients/dropout_1/div_grad/Sum_1Sum gradients/dropout_1/div_grad/mul4gradients/dropout_1/div_grad/BroadcastGradientArgs:1*
	keep_dims( *
T0*
_output_shapes
:*

Tidx0
¬
&gradients/dropout_1/div_grad/Reshape_1Reshape"gradients/dropout_1/div_grad/Sum_1$gradients/dropout_1/div_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:

-gradients/dropout_1/div_grad/tuple/group_depsNoOp%^gradients/dropout_1/div_grad/Reshape'^gradients/dropout_1/div_grad/Reshape_1

5gradients/dropout_1/div_grad/tuple/control_dependencyIdentity$gradients/dropout_1/div_grad/Reshape.^gradients/dropout_1/div_grad/tuple/group_deps*
T0*'
_output_shapes
:’’’’’’’’’x*7
_class-
+)loc:@gradients/dropout_1/div_grad/Reshape
ł
7gradients/dropout_1/div_grad/tuple/control_dependency_1Identity&gradients/dropout_1/div_grad/Reshape_1.^gradients/dropout_1/div_grad/tuple/group_deps*
T0*
_output_shapes
:*9
_class/
-+loc:@gradients/dropout_1/div_grad/Reshape_1

gradients/Relu_1_grad/ReluGradReluGrad5gradients/dropout_1/div_grad/tuple/control_dependencyRelu_1*
T0*'
_output_shapes
:’’’’’’’’’x
b
gradients/Add_1_grad/ShapeShapeMatMul_1*
T0*
out_type0*
_output_shapes
:
f
gradients/Add_1_grad/Shape_1Const*
valueB:x*
dtype0*
_output_shapes
:
ŗ
*gradients/Add_1_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/Add_1_grad/Shapegradients/Add_1_grad/Shape_1*
T0*2
_output_shapes 
:’’’’’’’’’:’’’’’’’’’
«
gradients/Add_1_grad/SumSumgradients/Relu_1_grad/ReluGrad*gradients/Add_1_grad/BroadcastGradientArgs*
	keep_dims( *
T0*
_output_shapes
:*

Tidx0

gradients/Add_1_grad/ReshapeReshapegradients/Add_1_grad/Sumgradients/Add_1_grad/Shape*
T0*
Tshape0*'
_output_shapes
:’’’’’’’’’x
Æ
gradients/Add_1_grad/Sum_1Sumgradients/Relu_1_grad/ReluGrad,gradients/Add_1_grad/BroadcastGradientArgs:1*
	keep_dims( *
T0*
_output_shapes
:*

Tidx0

gradients/Add_1_grad/Reshape_1Reshapegradients/Add_1_grad/Sum_1gradients/Add_1_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:x
m
%gradients/Add_1_grad/tuple/group_depsNoOp^gradients/Add_1_grad/Reshape^gradients/Add_1_grad/Reshape_1
ā
-gradients/Add_1_grad/tuple/control_dependencyIdentitygradients/Add_1_grad/Reshape&^gradients/Add_1_grad/tuple/group_deps*
T0*'
_output_shapes
:’’’’’’’’’x*/
_class%
#!loc:@gradients/Add_1_grad/Reshape
Ū
/gradients/Add_1_grad/tuple/control_dependency_1Identitygradients/Add_1_grad/Reshape_1&^gradients/Add_1_grad/tuple/group_deps*
T0*
_output_shapes
:x*1
_class'
%#loc:@gradients/Add_1_grad/Reshape_1
Ą
gradients/MatMul_1_grad/MatMulMatMul-gradients/Add_1_grad/tuple/control_dependencyVariable_1/read*
T0*
transpose_b(*
transpose_a( *'
_output_shapes
:’’’’’’’’’x
µ
 gradients/MatMul_1_grad/MatMul_1MatMuldropout/mul-gradients/Add_1_grad/tuple/control_dependency*
T0*
transpose_b( *
transpose_a(*
_output_shapes

:xx
t
(gradients/MatMul_1_grad/tuple/group_depsNoOp^gradients/MatMul_1_grad/MatMul!^gradients/MatMul_1_grad/MatMul_1
ģ
0gradients/MatMul_1_grad/tuple/control_dependencyIdentitygradients/MatMul_1_grad/MatMul)^gradients/MatMul_1_grad/tuple/group_deps*
T0*'
_output_shapes
:’’’’’’’’’x*1
_class'
%#loc:@gradients/MatMul_1_grad/MatMul
é
2gradients/MatMul_1_grad/tuple/control_dependency_1Identity gradients/MatMul_1_grad/MatMul_1)^gradients/MatMul_1_grad/tuple/group_deps*
T0*
_output_shapes

:xx*3
_class)
'%loc:@gradients/MatMul_1_grad/MatMul_1
Ū
gradients/AddN_1AddN/gradients/Add_2_grad/tuple/control_dependency_1/gradients/Add_1_grad/tuple/control_dependency_1*
T0*
N*
_output_shapes
:x*1
_class'
%#loc:@gradients/Add_2_grad/Reshape_1
t
 gradients/dropout/mul_grad/ShapeShapedropout/div*
T0*
out_type0*#
_output_shapes
:’’’’’’’’’
x
"gradients/dropout/mul_grad/Shape_1Shapedropout/Floor*
T0*
out_type0*#
_output_shapes
:’’’’’’’’’
Ģ
0gradients/dropout/mul_grad/BroadcastGradientArgsBroadcastGradientArgs gradients/dropout/mul_grad/Shape"gradients/dropout/mul_grad/Shape_1*
T0*2
_output_shapes 
:’’’’’’’’’:’’’’’’’’’

gradients/dropout/mul_grad/MulMul0gradients/MatMul_1_grad/tuple/control_dependencydropout/Floor*
T0*
_output_shapes
:
·
gradients/dropout/mul_grad/SumSumgradients/dropout/mul_grad/Mul0gradients/dropout/mul_grad/BroadcastGradientArgs*
	keep_dims( *
T0*
_output_shapes
:*

Tidx0
 
"gradients/dropout/mul_grad/ReshapeReshapegradients/dropout/mul_grad/Sum gradients/dropout/mul_grad/Shape*
T0*
Tshape0*
_output_shapes
:

 gradients/dropout/mul_grad/Mul_1Muldropout/div0gradients/MatMul_1_grad/tuple/control_dependency*
T0*
_output_shapes
:
½
 gradients/dropout/mul_grad/Sum_1Sum gradients/dropout/mul_grad/Mul_12gradients/dropout/mul_grad/BroadcastGradientArgs:1*
	keep_dims( *
T0*
_output_shapes
:*

Tidx0
¦
$gradients/dropout/mul_grad/Reshape_1Reshape gradients/dropout/mul_grad/Sum_1"gradients/dropout/mul_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:

+gradients/dropout/mul_grad/tuple/group_depsNoOp#^gradients/dropout/mul_grad/Reshape%^gradients/dropout/mul_grad/Reshape_1
ė
3gradients/dropout/mul_grad/tuple/control_dependencyIdentity"gradients/dropout/mul_grad/Reshape,^gradients/dropout/mul_grad/tuple/group_deps*
T0*
_output_shapes
:*5
_class+
)'loc:@gradients/dropout/mul_grad/Reshape
ń
5gradients/dropout/mul_grad/tuple/control_dependency_1Identity$gradients/dropout/mul_grad/Reshape_1,^gradients/dropout/mul_grad/tuple/group_deps*
T0*
_output_shapes
:*7
_class-
+)loc:@gradients/dropout/mul_grad/Reshape_1
ē
gradients/AddN_2AddN2gradients/MatMul_2_grad/tuple/control_dependency_12gradients/MatMul_1_grad/tuple/control_dependency_1*
T0*
N*
_output_shapes

:xx*3
_class)
'%loc:@gradients/MatMul_2_grad/MatMul_1
d
 gradients/dropout/div_grad/ShapeShapeRelu*
T0*
out_type0*
_output_shapes
:
t
"gradients/dropout/div_grad/Shape_1Shape	keep_prob*
T0*
out_type0*#
_output_shapes
:’’’’’’’’’
Ģ
0gradients/dropout/div_grad/BroadcastGradientArgsBroadcastGradientArgs gradients/dropout/div_grad/Shape"gradients/dropout/div_grad/Shape_1*
T0*2
_output_shapes 
:’’’’’’’’’:’’’’’’’’’

"gradients/dropout/div_grad/RealDivRealDiv3gradients/dropout/mul_grad/tuple/control_dependency	keep_prob*
T0*
_output_shapes
:
»
gradients/dropout/div_grad/SumSum"gradients/dropout/div_grad/RealDiv0gradients/dropout/div_grad/BroadcastGradientArgs*
	keep_dims( *
T0*
_output_shapes
:*

Tidx0
Æ
"gradients/dropout/div_grad/ReshapeReshapegradients/dropout/div_grad/Sum gradients/dropout/div_grad/Shape*
T0*
Tshape0*'
_output_shapes
:’’’’’’’’’x
]
gradients/dropout/div_grad/NegNegRelu*
T0*'
_output_shapes
:’’’’’’’’’x
}
$gradients/dropout/div_grad/RealDiv_1RealDivgradients/dropout/div_grad/Neg	keep_prob*
T0*
_output_shapes
:

$gradients/dropout/div_grad/RealDiv_2RealDiv$gradients/dropout/div_grad/RealDiv_1	keep_prob*
T0*
_output_shapes
:
£
gradients/dropout/div_grad/mulMul3gradients/dropout/mul_grad/tuple/control_dependency$gradients/dropout/div_grad/RealDiv_2*
T0*
_output_shapes
:
»
 gradients/dropout/div_grad/Sum_1Sumgradients/dropout/div_grad/mul2gradients/dropout/div_grad/BroadcastGradientArgs:1*
	keep_dims( *
T0*
_output_shapes
:*

Tidx0
¦
$gradients/dropout/div_grad/Reshape_1Reshape gradients/dropout/div_grad/Sum_1"gradients/dropout/div_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:

+gradients/dropout/div_grad/tuple/group_depsNoOp#^gradients/dropout/div_grad/Reshape%^gradients/dropout/div_grad/Reshape_1
ś
3gradients/dropout/div_grad/tuple/control_dependencyIdentity"gradients/dropout/div_grad/Reshape,^gradients/dropout/div_grad/tuple/group_deps*
T0*'
_output_shapes
:’’’’’’’’’x*5
_class+
)'loc:@gradients/dropout/div_grad/Reshape
ń
5gradients/dropout/div_grad/tuple/control_dependency_1Identity$gradients/dropout/div_grad/Reshape_1,^gradients/dropout/div_grad/tuple/group_deps*
T0*
_output_shapes
:*7
_class-
+)loc:@gradients/dropout/div_grad/Reshape_1

gradients/Relu_grad/ReluGradReluGrad3gradients/dropout/div_grad/tuple/control_dependencyRelu*
T0*'
_output_shapes
:’’’’’’’’’x
^
gradients/Add_grad/ShapeShapeMatMul*
T0*
out_type0*
_output_shapes
:
d
gradients/Add_grad/Shape_1Const*
valueB:x*
dtype0*
_output_shapes
:
“
(gradients/Add_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/Add_grad/Shapegradients/Add_grad/Shape_1*
T0*2
_output_shapes 
:’’’’’’’’’:’’’’’’’’’
„
gradients/Add_grad/SumSumgradients/Relu_grad/ReluGrad(gradients/Add_grad/BroadcastGradientArgs*
	keep_dims( *
T0*
_output_shapes
:*

Tidx0

gradients/Add_grad/ReshapeReshapegradients/Add_grad/Sumgradients/Add_grad/Shape*
T0*
Tshape0*'
_output_shapes
:’’’’’’’’’x
©
gradients/Add_grad/Sum_1Sumgradients/Relu_grad/ReluGrad*gradients/Add_grad/BroadcastGradientArgs:1*
	keep_dims( *
T0*
_output_shapes
:*

Tidx0

gradients/Add_grad/Reshape_1Reshapegradients/Add_grad/Sum_1gradients/Add_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:x
g
#gradients/Add_grad/tuple/group_depsNoOp^gradients/Add_grad/Reshape^gradients/Add_grad/Reshape_1
Ś
+gradients/Add_grad/tuple/control_dependencyIdentitygradients/Add_grad/Reshape$^gradients/Add_grad/tuple/group_deps*
T0*'
_output_shapes
:’’’’’’’’’x*-
_class#
!loc:@gradients/Add_grad/Reshape
Ó
-gradients/Add_grad/tuple/control_dependency_1Identitygradients/Add_grad/Reshape_1$^gradients/Add_grad/tuple/group_deps*
T0*
_output_shapes
:x*/
_class%
#!loc:@gradients/Add_grad/Reshape_1
ŗ
gradients/MatMul_grad/MatMulMatMul+gradients/Add_grad/tuple/control_dependencyVariable/read*
T0*
transpose_b(*
transpose_a( *'
_output_shapes
:’’’’’’’’’	
Æ
gradients/MatMul_grad/MatMul_1MatMul	inputs_ph+gradients/Add_grad/tuple/control_dependency*
T0*
transpose_b( *
transpose_a(*
_output_shapes

:	x
n
&gradients/MatMul_grad/tuple/group_depsNoOp^gradients/MatMul_grad/MatMul^gradients/MatMul_grad/MatMul_1
ä
.gradients/MatMul_grad/tuple/control_dependencyIdentitygradients/MatMul_grad/MatMul'^gradients/MatMul_grad/tuple/group_deps*
T0*'
_output_shapes
:’’’’’’’’’	*/
_class%
#!loc:@gradients/MatMul_grad/MatMul
į
0gradients/MatMul_grad/tuple/control_dependency_1Identitygradients/MatMul_grad/MatMul_1'^gradients/MatMul_grad/tuple/group_deps*
T0*
_output_shapes

:	x*1
_class'
%#loc:@gradients/MatMul_grad/MatMul_1
{
beta1_power/initial_valueConst*
valueB
 *fff?*
dtype0*
_output_shapes
: *
_class
loc:@Variable

beta1_power
VariableV2*
shared_name *
_class
loc:@Variable*
_output_shapes
: *
dtype0*
shape: *
	container 
«
beta1_power/AssignAssignbeta1_powerbeta1_power/initial_value*
T0*
_output_shapes
: *
use_locking(*
validate_shape(*
_class
loc:@Variable
g
beta1_power/readIdentitybeta1_power*
T0*
_output_shapes
: *
_class
loc:@Variable
{
beta2_power/initial_valueConst*
valueB
 *w¾?*
dtype0*
_output_shapes
: *
_class
loc:@Variable

beta2_power
VariableV2*
shared_name *
_class
loc:@Variable*
_output_shapes
: *
dtype0*
shape: *
	container 
«
beta2_power/AssignAssignbeta2_powerbeta2_power/initial_value*
T0*
_output_shapes
: *
use_locking(*
validate_shape(*
_class
loc:@Variable
g
beta2_power/readIdentitybeta2_power*
T0*
_output_shapes
: *
_class
loc:@Variable

/Variable/Adam/Initializer/zeros/shape_as_tensorConst*
valueB"	   x   *
dtype0*
_output_shapes
:*
_class
loc:@Variable

%Variable/Adam/Initializer/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: *
_class
loc:@Variable
×
Variable/Adam/Initializer/zerosFill/Variable/Adam/Initializer/zeros/shape_as_tensor%Variable/Adam/Initializer/zeros/Const*
T0*
_output_shapes

:	x*

index_type0*
_class
loc:@Variable

Variable/Adam
VariableV2*
shared_name *
_class
loc:@Variable*
_output_shapes

:	x*
dtype0*
shape
:	x*
	container 
½
Variable/Adam/AssignAssignVariable/AdamVariable/Adam/Initializer/zeros*
T0*
_output_shapes

:	x*
use_locking(*
validate_shape(*
_class
loc:@Variable
s
Variable/Adam/readIdentityVariable/Adam*
T0*
_output_shapes

:	x*
_class
loc:@Variable

1Variable/Adam_1/Initializer/zeros/shape_as_tensorConst*
valueB"	   x   *
dtype0*
_output_shapes
:*
_class
loc:@Variable

'Variable/Adam_1/Initializer/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: *
_class
loc:@Variable
Ż
!Variable/Adam_1/Initializer/zerosFill1Variable/Adam_1/Initializer/zeros/shape_as_tensor'Variable/Adam_1/Initializer/zeros/Const*
T0*
_output_shapes

:	x*

index_type0*
_class
loc:@Variable
 
Variable/Adam_1
VariableV2*
shared_name *
_class
loc:@Variable*
_output_shapes

:	x*
dtype0*
shape
:	x*
	container 
Ć
Variable/Adam_1/AssignAssignVariable/Adam_1!Variable/Adam_1/Initializer/zeros*
T0*
_output_shapes

:	x*
use_locking(*
validate_shape(*
_class
loc:@Variable
w
Variable/Adam_1/readIdentityVariable/Adam_1*
T0*
_output_shapes

:	x*
_class
loc:@Variable
”
1Variable_1/Adam/Initializer/zeros/shape_as_tensorConst*
valueB"x   x   *
dtype0*
_output_shapes
:*
_class
loc:@Variable_1

'Variable_1/Adam/Initializer/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: *
_class
loc:@Variable_1
ß
!Variable_1/Adam/Initializer/zerosFill1Variable_1/Adam/Initializer/zeros/shape_as_tensor'Variable_1/Adam/Initializer/zeros/Const*
T0*
_output_shapes

:xx*

index_type0*
_class
loc:@Variable_1
¢
Variable_1/Adam
VariableV2*
shared_name *
_class
loc:@Variable_1*
_output_shapes

:xx*
dtype0*
shape
:xx*
	container 
Å
Variable_1/Adam/AssignAssignVariable_1/Adam!Variable_1/Adam/Initializer/zeros*
T0*
_output_shapes

:xx*
use_locking(*
validate_shape(*
_class
loc:@Variable_1
y
Variable_1/Adam/readIdentityVariable_1/Adam*
T0*
_output_shapes

:xx*
_class
loc:@Variable_1
£
3Variable_1/Adam_1/Initializer/zeros/shape_as_tensorConst*
valueB"x   x   *
dtype0*
_output_shapes
:*
_class
loc:@Variable_1

)Variable_1/Adam_1/Initializer/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: *
_class
loc:@Variable_1
å
#Variable_1/Adam_1/Initializer/zerosFill3Variable_1/Adam_1/Initializer/zeros/shape_as_tensor)Variable_1/Adam_1/Initializer/zeros/Const*
T0*
_output_shapes

:xx*

index_type0*
_class
loc:@Variable_1
¤
Variable_1/Adam_1
VariableV2*
shared_name *
_class
loc:@Variable_1*
_output_shapes

:xx*
dtype0*
shape
:xx*
	container 
Ė
Variable_1/Adam_1/AssignAssignVariable_1/Adam_1#Variable_1/Adam_1/Initializer/zeros*
T0*
_output_shapes

:xx*
use_locking(*
validate_shape(*
_class
loc:@Variable_1
}
Variable_1/Adam_1/readIdentityVariable_1/Adam_1*
T0*
_output_shapes

:xx*
_class
loc:@Variable_1

!Variable_3/Adam/Initializer/zerosConst*
valueBx*    *
dtype0*
_output_shapes

:x*
_class
loc:@Variable_3
¢
Variable_3/Adam
VariableV2*
shared_name *
_class
loc:@Variable_3*
_output_shapes

:x*
dtype0*
shape
:x*
	container 
Å
Variable_3/Adam/AssignAssignVariable_3/Adam!Variable_3/Adam/Initializer/zeros*
T0*
_output_shapes

:x*
use_locking(*
validate_shape(*
_class
loc:@Variable_3
y
Variable_3/Adam/readIdentityVariable_3/Adam*
T0*
_output_shapes

:x*
_class
loc:@Variable_3

#Variable_3/Adam_1/Initializer/zerosConst*
valueBx*    *
dtype0*
_output_shapes

:x*
_class
loc:@Variable_3
¤
Variable_3/Adam_1
VariableV2*
shared_name *
_class
loc:@Variable_3*
_output_shapes

:x*
dtype0*
shape
:x*
	container 
Ė
Variable_3/Adam_1/AssignAssignVariable_3/Adam_1#Variable_3/Adam_1/Initializer/zeros*
T0*
_output_shapes

:x*
use_locking(*
validate_shape(*
_class
loc:@Variable_3
}
Variable_3/Adam_1/readIdentityVariable_3/Adam_1*
T0*
_output_shapes

:x*
_class
loc:@Variable_3

!Variable_4/Adam/Initializer/zerosConst*
valueBx*    *
dtype0*
_output_shapes
:x*
_class
loc:@Variable_4

Variable_4/Adam
VariableV2*
shared_name *
_class
loc:@Variable_4*
_output_shapes
:x*
dtype0*
shape:x*
	container 
Į
Variable_4/Adam/AssignAssignVariable_4/Adam!Variable_4/Adam/Initializer/zeros*
T0*
_output_shapes
:x*
use_locking(*
validate_shape(*
_class
loc:@Variable_4
u
Variable_4/Adam/readIdentityVariable_4/Adam*
T0*
_output_shapes
:x*
_class
loc:@Variable_4

#Variable_4/Adam_1/Initializer/zerosConst*
valueBx*    *
dtype0*
_output_shapes
:x*
_class
loc:@Variable_4

Variable_4/Adam_1
VariableV2*
shared_name *
_class
loc:@Variable_4*
_output_shapes
:x*
dtype0*
shape:x*
	container 
Ē
Variable_4/Adam_1/AssignAssignVariable_4/Adam_1#Variable_4/Adam_1/Initializer/zeros*
T0*
_output_shapes
:x*
use_locking(*
validate_shape(*
_class
loc:@Variable_4
y
Variable_4/Adam_1/readIdentityVariable_4/Adam_1*
T0*
_output_shapes
:x*
_class
loc:@Variable_4

!Variable_5/Adam/Initializer/zerosConst*
valueBx*    *
dtype0*
_output_shapes
:x*
_class
loc:@Variable_5

Variable_5/Adam
VariableV2*
shared_name *
_class
loc:@Variable_5*
_output_shapes
:x*
dtype0*
shape:x*
	container 
Į
Variable_5/Adam/AssignAssignVariable_5/Adam!Variable_5/Adam/Initializer/zeros*
T0*
_output_shapes
:x*
use_locking(*
validate_shape(*
_class
loc:@Variable_5
u
Variable_5/Adam/readIdentityVariable_5/Adam*
T0*
_output_shapes
:x*
_class
loc:@Variable_5

#Variable_5/Adam_1/Initializer/zerosConst*
valueBx*    *
dtype0*
_output_shapes
:x*
_class
loc:@Variable_5

Variable_5/Adam_1
VariableV2*
shared_name *
_class
loc:@Variable_5*
_output_shapes
:x*
dtype0*
shape:x*
	container 
Ē
Variable_5/Adam_1/AssignAssignVariable_5/Adam_1#Variable_5/Adam_1/Initializer/zeros*
T0*
_output_shapes
:x*
use_locking(*
validate_shape(*
_class
loc:@Variable_5
y
Variable_5/Adam_1/readIdentityVariable_5/Adam_1*
T0*
_output_shapes
:x*
_class
loc:@Variable_5

!Variable_7/Adam/Initializer/zerosConst*
valueB*    *
dtype0*
_output_shapes
:*
_class
loc:@Variable_7

Variable_7/Adam
VariableV2*
shared_name *
_class
loc:@Variable_7*
_output_shapes
:*
dtype0*
shape:*
	container 
Į
Variable_7/Adam/AssignAssignVariable_7/Adam!Variable_7/Adam/Initializer/zeros*
T0*
_output_shapes
:*
use_locking(*
validate_shape(*
_class
loc:@Variable_7
u
Variable_7/Adam/readIdentityVariable_7/Adam*
T0*
_output_shapes
:*
_class
loc:@Variable_7

#Variable_7/Adam_1/Initializer/zerosConst*
valueB*    *
dtype0*
_output_shapes
:*
_class
loc:@Variable_7

Variable_7/Adam_1
VariableV2*
shared_name *
_class
loc:@Variable_7*
_output_shapes
:*
dtype0*
shape:*
	container 
Ē
Variable_7/Adam_1/AssignAssignVariable_7/Adam_1#Variable_7/Adam_1/Initializer/zeros*
T0*
_output_shapes
:*
use_locking(*
validate_shape(*
_class
loc:@Variable_7
y
Variable_7/Adam_1/readIdentityVariable_7/Adam_1*
T0*
_output_shapes
:*
_class
loc:@Variable_7
W
Adam/learning_rateConst*
valueB
 *RI9*
dtype0*
_output_shapes
: 
O

Adam/beta1Const*
valueB
 *fff?*
dtype0*
_output_shapes
: 
O

Adam/beta2Const*
valueB
 *w¾?*
dtype0*
_output_shapes
: 
Q
Adam/epsilonConst*
valueB
 *wĢ+2*
dtype0*
_output_shapes
: 
Ņ
Adam/update_Variable/ApplyAdam	ApplyAdamVariableVariable/AdamVariable/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon0gradients/MatMul_grad/tuple/control_dependency_1*
T0*
_output_shapes

:	x*
use_locking( *
use_nesterov( *
_class
loc:@Variable
¼
 Adam/update_Variable_1/ApplyAdam	ApplyAdam
Variable_1Variable_1/AdamVariable_1/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilongradients/AddN_2*
T0*
_output_shapes

:xx*
use_locking( *
use_nesterov( *
_class
loc:@Variable_1
Ž
 Adam/update_Variable_3/ApplyAdam	ApplyAdam
Variable_3Variable_3/AdamVariable_3/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon2gradients/MatMul_3_grad/tuple/control_dependency_1*
T0*
_output_shapes

:x*
use_locking( *
use_nesterov( *
_class
loc:@Variable_3
Õ
 Adam/update_Variable_4/ApplyAdam	ApplyAdam
Variable_4Variable_4/AdamVariable_4/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon-gradients/Add_grad/tuple/control_dependency_1*
T0*
_output_shapes
:x*
use_locking( *
use_nesterov( *
_class
loc:@Variable_4
ø
 Adam/update_Variable_5/ApplyAdam	ApplyAdam
Variable_5Variable_5/AdamVariable_5/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilongradients/AddN_1*
T0*
_output_shapes
:x*
use_locking( *
use_nesterov( *
_class
loc:@Variable_5
×
 Adam/update_Variable_7/ApplyAdam	ApplyAdam
Variable_7Variable_7/AdamVariable_7/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon/gradients/add_3_grad/tuple/control_dependency_1*
T0*
_output_shapes
:*
use_locking( *
use_nesterov( *
_class
loc:@Variable_7
»
Adam/mulMulbeta1_power/read
Adam/beta1^Adam/update_Variable/ApplyAdam!^Adam/update_Variable_1/ApplyAdam!^Adam/update_Variable_3/ApplyAdam!^Adam/update_Variable_4/ApplyAdam!^Adam/update_Variable_5/ApplyAdam!^Adam/update_Variable_7/ApplyAdam*
T0*
_output_shapes
: *
_class
loc:@Variable

Adam/AssignAssignbeta1_powerAdam/mul*
T0*
_output_shapes
: *
use_locking( *
validate_shape(*
_class
loc:@Variable
½

Adam/mul_1Mulbeta2_power/read
Adam/beta2^Adam/update_Variable/ApplyAdam!^Adam/update_Variable_1/ApplyAdam!^Adam/update_Variable_3/ApplyAdam!^Adam/update_Variable_4/ApplyAdam!^Adam/update_Variable_5/ApplyAdam!^Adam/update_Variable_7/ApplyAdam*
T0*
_output_shapes
: *
_class
loc:@Variable

Adam/Assign_1Assignbeta2_power
Adam/mul_1*
T0*
_output_shapes
: *
use_locking( *
validate_shape(*
_class
loc:@Variable
ś
AdamNoOp^Adam/Assign^Adam/Assign_1^Adam/update_Variable/ApplyAdam!^Adam/update_Variable_1/ApplyAdam!^Adam/update_Variable_3/ApplyAdam!^Adam/update_Variable_4/ApplyAdam!^Adam/update_Variable_5/ApplyAdam!^Adam/update_Variable_7/ApplyAdam
K
SigmoidSigmoidadd_3*
T0*'
_output_shapes
:’’’’’’’’’
N

predictionRoundSigmoid*
T0*'
_output_shapes
:’’’’’’’’’
W
EqualEqual
prediction	labels_ph*
T0*'
_output_shapes
:’’’’’’’’’
d
CastCastEqual*

DstT0*'
_output_shapes
:’’’’’’’’’*
Truncate( *

SrcT0

p
recall/CastCast
prediction*

DstT0
*'
_output_shapes
:’’’’’’’’’*
Truncate( *

SrcT0
q
recall/Cast_1Cast	labels_ph*

DstT0
*'
_output_shapes
:’’’’’’’’’*
Truncate( *

SrcT0
_
recall/true_positives/Equal/yConst*
value	B
 Z*
dtype0
*
_output_shapes
: 

recall/true_positives/EqualEqualrecall/Cast_1recall/true_positives/Equal/y*
T0
*'
_output_shapes
:’’’’’’’’’
a
recall/true_positives/Equal_1/yConst*
value	B
 Z*
dtype0
*
_output_shapes
: 

recall/true_positives/Equal_1Equalrecall/Castrecall/true_positives/Equal_1/y*
T0
*'
_output_shapes
:’’’’’’’’’

 recall/true_positives/LogicalAnd
LogicalAndrecall/true_positives/Equalrecall/true_positives/Equal_1*'
_output_shapes
:’’’’’’’’’
L
Drecall/true_positives/assert_type/statically_determined_correct_typeNoOp
¢
-recall/true_positives/count/Initializer/zerosConst*
valueB
 *    *
dtype0*
_output_shapes
: *.
_class$
" loc:@recall/true_positives/count
Æ
recall/true_positives/count
VariableV2*
shared_name *.
_class$
" loc:@recall/true_positives/count*
_output_shapes
: *
dtype0*
shape: *
	container 
ņ
"recall/true_positives/count/AssignAssignrecall/true_positives/count-recall/true_positives/count/Initializer/zeros*
T0*
_output_shapes
: *
use_locking(*
validate_shape(*.
_class$
" loc:@recall/true_positives/count

 recall/true_positives/count/readIdentityrecall/true_positives/count*
T0*
_output_shapes
: *.
_class$
" loc:@recall/true_positives/count

recall/true_positives/ToFloatCast recall/true_positives/LogicalAnd*

DstT0*'
_output_shapes
:’’’’’’’’’*
Truncate( *

SrcT0

m
recall/true_positives/IdentityIdentity recall/true_positives/count/read*
T0*
_output_shapes
: 
l
recall/true_positives/ConstConst*
valueB"       *
dtype0*
_output_shapes
:

recall/true_positives/SumSumrecall/true_positives/ToFloatrecall/true_positives/Const*
	keep_dims( *
T0*
_output_shapes
: *

Tidx0
Č
recall/true_positives/AssignAdd	AssignAddrecall/true_positives/countrecall/true_positives/Sum*
T0*
_output_shapes
: *
use_locking( *.
_class$
" loc:@recall/true_positives/count
`
recall/false_negatives/Equal/yConst*
value	B
 Z*
dtype0
*
_output_shapes
: 

recall/false_negatives/EqualEqualrecall/Cast_1recall/false_negatives/Equal/y*
T0
*'
_output_shapes
:’’’’’’’’’
b
 recall/false_negatives/Equal_1/yConst*
value	B
 Z *
dtype0
*
_output_shapes
: 

recall/false_negatives/Equal_1Equalrecall/Cast recall/false_negatives/Equal_1/y*
T0
*'
_output_shapes
:’’’’’’’’’

!recall/false_negatives/LogicalAnd
LogicalAndrecall/false_negatives/Equalrecall/false_negatives/Equal_1*'
_output_shapes
:’’’’’’’’’
M
Erecall/false_negatives/assert_type/statically_determined_correct_typeNoOp
¤
.recall/false_negatives/count/Initializer/zerosConst*
valueB
 *    *
dtype0*
_output_shapes
: */
_class%
#!loc:@recall/false_negatives/count
±
recall/false_negatives/count
VariableV2*
shared_name */
_class%
#!loc:@recall/false_negatives/count*
_output_shapes
: *
dtype0*
shape: *
	container 
ö
#recall/false_negatives/count/AssignAssignrecall/false_negatives/count.recall/false_negatives/count/Initializer/zeros*
T0*
_output_shapes
: *
use_locking(*
validate_shape(*/
_class%
#!loc:@recall/false_negatives/count

!recall/false_negatives/count/readIdentityrecall/false_negatives/count*
T0*
_output_shapes
: */
_class%
#!loc:@recall/false_negatives/count

recall/false_negatives/ToFloatCast!recall/false_negatives/LogicalAnd*

DstT0*'
_output_shapes
:’’’’’’’’’*
Truncate( *

SrcT0

o
recall/false_negatives/IdentityIdentity!recall/false_negatives/count/read*
T0*
_output_shapes
: 
m
recall/false_negatives/ConstConst*
valueB"       *
dtype0*
_output_shapes
:

recall/false_negatives/SumSumrecall/false_negatives/ToFloatrecall/false_negatives/Const*
	keep_dims( *
T0*
_output_shapes
: *

Tidx0
Ģ
 recall/false_negatives/AssignAdd	AssignAddrecall/false_negatives/countrecall/false_negatives/Sum*
T0*
_output_shapes
: *
use_locking( */
_class%
#!loc:@recall/false_negatives/count
s

recall/addAddrecall/true_positives/Identityrecall/false_negatives/Identity*
T0*
_output_shapes
: 
U
recall/Greater/yConst*
valueB
 *    *
dtype0*
_output_shapes
: 
X
recall/GreaterGreater
recall/addrecall/Greater/y*
T0*
_output_shapes
: 
u
recall/add_1Addrecall/true_positives/Identityrecall/false_negatives/Identity*
T0*
_output_shapes
: 
d

recall/divRealDivrecall/true_positives/Identityrecall/add_1*
T0*
_output_shapes
: 
S
recall/value/eConst*
valueB
 *    *
dtype0*
_output_shapes
: 
c
recall/valueSelectrecall/Greater
recall/divrecall/value/e*
T0*
_output_shapes
: 
w
recall/add_2Addrecall/true_positives/AssignAdd recall/false_negatives/AssignAdd*
T0*
_output_shapes
: 
W
recall/Greater_1/yConst*
valueB
 *    *
dtype0*
_output_shapes
: 
^
recall/Greater_1Greaterrecall/add_2recall/Greater_1/y*
T0*
_output_shapes
: 
w
recall/add_3Addrecall/true_positives/AssignAdd recall/false_negatives/AssignAdd*
T0*
_output_shapes
: 
g
recall/div_1RealDivrecall/true_positives/AssignAddrecall/add_3*
T0*
_output_shapes
: 
W
recall/update_op/eConst*
valueB
 *    *
dtype0*
_output_shapes
: 
o
recall/update_opSelectrecall/Greater_1recall/div_1recall/update_op/e*
T0*
_output_shapes
: 
X
Const_1Const*
valueB"       *
dtype0*
_output_shapes
:
[
Mean_1MeanCastConst_1*
	keep_dims( *
T0*
_output_shapes
: *

Tidx0

initNoOp^Variable/Adam/Assign^Variable/Adam_1/Assign^Variable/Assign^Variable_1/Adam/Assign^Variable_1/Adam_1/Assign^Variable_1/Assign^Variable_2/Assign^Variable_3/Adam/Assign^Variable_3/Adam_1/Assign^Variable_3/Assign^Variable_4/Adam/Assign^Variable_4/Adam_1/Assign^Variable_4/Assign^Variable_5/Adam/Assign^Variable_5/Adam_1/Assign^Variable_5/Assign^Variable_6/Assign^Variable_7/Adam/Assign^Variable_7/Adam_1/Assign^Variable_7/Assign^beta1_power/Assign^beta2_power/Assign
P

save/ConstConst*
valueB Bmodel*
dtype0*
_output_shapes
: 

save/StringJoin/inputs_1Const*<
value3B1 B+_temp_514b9bbc633e4226a72aaf88e6a5c39e/part*
dtype0*
_output_shapes
: 
u
save/StringJoin
StringJoin
save/Constsave/StringJoin/inputs_1*
N*
	separator *
_output_shapes
: 
Q
save/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
\
save/ShardedFilename/shardConst*
value	B : *
dtype0*
_output_shapes
: 
}
save/ShardedFilenameShardedFilenamesave/StringJoinsave/ShardedFilename/shardsave/num_shards*
_output_shapes
: 
®
save/SaveV2/tensor_namesConst*į
value×BŌBVariableBVariable/AdamBVariable/Adam_1B
Variable_1BVariable_1/AdamBVariable_1/Adam_1B
Variable_2B
Variable_3BVariable_3/AdamBVariable_3/Adam_1B
Variable_4BVariable_4/AdamBVariable_4/Adam_1B
Variable_5BVariable_5/AdamBVariable_5/Adam_1B
Variable_6B
Variable_7BVariable_7/AdamBVariable_7/Adam_1Bbeta1_powerBbeta2_power*
dtype0*
_output_shapes
:

save/SaveV2/shape_and_slicesConst*?
value6B4B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:
Õ
save/SaveV2SaveV2save/ShardedFilenamesave/SaveV2/tensor_namessave/SaveV2/shape_and_slicesVariableVariable/AdamVariable/Adam_1
Variable_1Variable_1/AdamVariable_1/Adam_1
Variable_2
Variable_3Variable_3/AdamVariable_3/Adam_1
Variable_4Variable_4/AdamVariable_4/Adam_1
Variable_5Variable_5/AdamVariable_5/Adam_1
Variable_6
Variable_7Variable_7/AdamVariable_7/Adam_1beta1_powerbeta2_power*$
dtypes
2

save/control_dependencyIdentitysave/ShardedFilename^save/SaveV2*
T0*
_output_shapes
: *'
_class
loc:@save/ShardedFilename

+save/MergeV2Checkpoints/checkpoint_prefixesPacksave/ShardedFilename^save/control_dependency*

axis *
T0*
N*
_output_shapes
:
}
save/MergeV2CheckpointsMergeV2Checkpoints+save/MergeV2Checkpoints/checkpoint_prefixes
save/Const*
delete_old_dirs(
z
save/IdentityIdentity
save/Const^save/MergeV2Checkpoints^save/control_dependency*
T0*
_output_shapes
: 
±
save/RestoreV2/tensor_namesConst*į
value×BŌBVariableBVariable/AdamBVariable/Adam_1B
Variable_1BVariable_1/AdamBVariable_1/Adam_1B
Variable_2B
Variable_3BVariable_3/AdamBVariable_3/Adam_1B
Variable_4BVariable_4/AdamBVariable_4/Adam_1B
Variable_5BVariable_5/AdamBVariable_5/Adam_1B
Variable_6B
Variable_7BVariable_7/AdamBVariable_7/Adam_1Bbeta1_powerBbeta2_power*
dtype0*
_output_shapes
:

save/RestoreV2/shape_and_slicesConst*?
value6B4B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:
ł
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices*$
dtypes
2*l
_output_shapesZ
X::::::::::::::::::::::

save/AssignAssignVariablesave/RestoreV2*
T0*
_output_shapes

:	x*
use_locking(*
validate_shape(*
_class
loc:@Variable
§
save/Assign_1AssignVariable/Adamsave/RestoreV2:1*
T0*
_output_shapes

:	x*
use_locking(*
validate_shape(*
_class
loc:@Variable
©
save/Assign_2AssignVariable/Adam_1save/RestoreV2:2*
T0*
_output_shapes

:	x*
use_locking(*
validate_shape(*
_class
loc:@Variable
¦
save/Assign_3Assign
Variable_1save/RestoreV2:3*
T0*
_output_shapes

:xx*
use_locking(*
validate_shape(*
_class
loc:@Variable_1
«
save/Assign_4AssignVariable_1/Adamsave/RestoreV2:4*
T0*
_output_shapes

:xx*
use_locking(*
validate_shape(*
_class
loc:@Variable_1
­
save/Assign_5AssignVariable_1/Adam_1save/RestoreV2:5*
T0*
_output_shapes

:xx*
use_locking(*
validate_shape(*
_class
loc:@Variable_1
¦
save/Assign_6Assign
Variable_2save/RestoreV2:6*
T0*
_output_shapes

:xx*
use_locking(*
validate_shape(*
_class
loc:@Variable_2
¦
save/Assign_7Assign
Variable_3save/RestoreV2:7*
T0*
_output_shapes

:x*
use_locking(*
validate_shape(*
_class
loc:@Variable_3
«
save/Assign_8AssignVariable_3/Adamsave/RestoreV2:8*
T0*
_output_shapes

:x*
use_locking(*
validate_shape(*
_class
loc:@Variable_3
­
save/Assign_9AssignVariable_3/Adam_1save/RestoreV2:9*
T0*
_output_shapes

:x*
use_locking(*
validate_shape(*
_class
loc:@Variable_3
¤
save/Assign_10Assign
Variable_4save/RestoreV2:10*
T0*
_output_shapes
:x*
use_locking(*
validate_shape(*
_class
loc:@Variable_4
©
save/Assign_11AssignVariable_4/Adamsave/RestoreV2:11*
T0*
_output_shapes
:x*
use_locking(*
validate_shape(*
_class
loc:@Variable_4
«
save/Assign_12AssignVariable_4/Adam_1save/RestoreV2:12*
T0*
_output_shapes
:x*
use_locking(*
validate_shape(*
_class
loc:@Variable_4
¤
save/Assign_13Assign
Variable_5save/RestoreV2:13*
T0*
_output_shapes
:x*
use_locking(*
validate_shape(*
_class
loc:@Variable_5
©
save/Assign_14AssignVariable_5/Adamsave/RestoreV2:14*
T0*
_output_shapes
:x*
use_locking(*
validate_shape(*
_class
loc:@Variable_5
«
save/Assign_15AssignVariable_5/Adam_1save/RestoreV2:15*
T0*
_output_shapes
:x*
use_locking(*
validate_shape(*
_class
loc:@Variable_5
¤
save/Assign_16Assign
Variable_6save/RestoreV2:16*
T0*
_output_shapes
:x*
use_locking(*
validate_shape(*
_class
loc:@Variable_6
¤
save/Assign_17Assign
Variable_7save/RestoreV2:17*
T0*
_output_shapes
:*
use_locking(*
validate_shape(*
_class
loc:@Variable_7
©
save/Assign_18AssignVariable_7/Adamsave/RestoreV2:18*
T0*
_output_shapes
:*
use_locking(*
validate_shape(*
_class
loc:@Variable_7
«
save/Assign_19AssignVariable_7/Adam_1save/RestoreV2:19*
T0*
_output_shapes
:*
use_locking(*
validate_shape(*
_class
loc:@Variable_7

save/Assign_20Assignbeta1_powersave/RestoreV2:20*
T0*
_output_shapes
: *
use_locking(*
validate_shape(*
_class
loc:@Variable

save/Assign_21Assignbeta2_powersave/RestoreV2:21*
T0*
_output_shapes
: *
use_locking(*
validate_shape(*
_class
loc:@Variable

save/restore_shardNoOp^save/Assign^save/Assign_1^save/Assign_10^save/Assign_11^save/Assign_12^save/Assign_13^save/Assign_14^save/Assign_15^save/Assign_16^save/Assign_17^save/Assign_18^save/Assign_19^save/Assign_2^save/Assign_20^save/Assign_21^save/Assign_3^save/Assign_4^save/Assign_5^save/Assign_6^save/Assign_7^save/Assign_8^save/Assign_9
-
save/restore_allNoOp^save/restore_shard "<
save/Const:0save/Identity:0save/restore_all (5 @F8"U
metric_variablesA
?
recall/true_positives/count:0
recall/false_negatives/count:0"ó
trainable_variablesŪŲ
B

Variable:0Variable/AssignVariable/read:02random_uniform:08
J
Variable_1:0Variable_1/AssignVariable_1/read:02random_uniform_1:08
J
Variable_2:0Variable_2/AssignVariable_2/read:02random_uniform_2:08
J
Variable_3:0Variable_3/AssignVariable_3/read:02random_uniform_3:08
J
Variable_4:0Variable_4/AssignVariable_4/read:02random_uniform_4:08
J
Variable_5:0Variable_5/AssignVariable_5/read:02random_uniform_5:08
J
Variable_6:0Variable_6/AssignVariable_6/read:02random_uniform_6:08
J
Variable_7:0Variable_7/AssignVariable_7/read:02random_uniform_7:08"Ń
local_variables½ŗ

recall/true_positives/count:0"recall/true_positives/count/Assign"recall/true_positives/count/read:02/recall/true_positives/count/Initializer/zeros:0

recall/false_negatives/count:0#recall/false_negatives/count/Assign#recall/false_negatives/count/read:020recall/false_negatives/count/Initializer/zeros:0"
train_op

Adam"­
	variables
B

Variable:0Variable/AssignVariable/read:02random_uniform:08
J
Variable_1:0Variable_1/AssignVariable_1/read:02random_uniform_1:08
J
Variable_2:0Variable_2/AssignVariable_2/read:02random_uniform_2:08
J
Variable_3:0Variable_3/AssignVariable_3/read:02random_uniform_3:08
J
Variable_4:0Variable_4/AssignVariable_4/read:02random_uniform_4:08
J
Variable_5:0Variable_5/AssignVariable_5/read:02random_uniform_5:08
J
Variable_6:0Variable_6/AssignVariable_6/read:02random_uniform_6:08
J
Variable_7:0Variable_7/AssignVariable_7/read:02random_uniform_7:08
T
beta1_power:0beta1_power/Assignbeta1_power/read:02beta1_power/initial_value:0
T
beta2_power:0beta2_power/Assignbeta2_power/read:02beta2_power/initial_value:0
`
Variable/Adam:0Variable/Adam/AssignVariable/Adam/read:02!Variable/Adam/Initializer/zeros:0
h
Variable/Adam_1:0Variable/Adam_1/AssignVariable/Adam_1/read:02#Variable/Adam_1/Initializer/zeros:0
h
Variable_1/Adam:0Variable_1/Adam/AssignVariable_1/Adam/read:02#Variable_1/Adam/Initializer/zeros:0
p
Variable_1/Adam_1:0Variable_1/Adam_1/AssignVariable_1/Adam_1/read:02%Variable_1/Adam_1/Initializer/zeros:0
h
Variable_3/Adam:0Variable_3/Adam/AssignVariable_3/Adam/read:02#Variable_3/Adam/Initializer/zeros:0
p
Variable_3/Adam_1:0Variable_3/Adam_1/AssignVariable_3/Adam_1/read:02%Variable_3/Adam_1/Initializer/zeros:0
h
Variable_4/Adam:0Variable_4/Adam/AssignVariable_4/Adam/read:02#Variable_4/Adam/Initializer/zeros:0
p
Variable_4/Adam_1:0Variable_4/Adam_1/AssignVariable_4/Adam_1/read:02%Variable_4/Adam_1/Initializer/zeros:0
h
Variable_5/Adam:0Variable_5/Adam/AssignVariable_5/Adam/read:02#Variable_5/Adam/Initializer/zeros:0
p
Variable_5/Adam_1:0Variable_5/Adam_1/AssignVariable_5/Adam_1/read:02%Variable_5/Adam_1/Initializer/zeros:0
h
Variable_7/Adam:0Variable_7/Adam/AssignVariable_7/Adam/read:02#Variable_7/Adam/Initializer/zeros:0
p
Variable_7/Adam_1:0Variable_7/Adam_1/AssignVariable_7/Adam_1/read:02%Variable_7/Adam_1/Initializer/zeros:0*ē
serving_defaultÓ
/
	labels_ph"
labels_ph:0’’’’’’’’’
 
	keep_prob
keep_prob:0
/
	inputs_ph"
inputs_ph:0’’’’’’’’’	1

prediction#
prediction:0’’’’’’’’’tensorflow/serving/predict