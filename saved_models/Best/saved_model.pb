�
��
B
AssignVariableOp
resource
value"dtype"
dtypetype�
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
�
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

�
Conv2DBackpropInput
input_sizes
filter"T
out_backprop"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

.
Identity

input"T
output"T"	
Ttype
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(�
=
Mul
x"T
y"T
z"T"
Ttype:
2	�
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
@
ReadVariableOp
resource
value"dtype"
dtypetype�
E
Relu
features"T
activations"T"
Ttype:
2	
�
ResizeNearestNeighbor
images"T
size
resized_images"T"
Ttype:
2
	"
align_cornersbool( "
half_pixel_centersbool( 
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
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
�
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring �
@
StaticRegexFullMatch	
input

output
"
patternstring
�
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
-
Tanh
x"T
y"T"
Ttype:

2
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.4.12v2.4.0-49-g85c8b2a817f8��
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
�
0bigger_auto_encoder/bigger_encoder/conv2d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*A
shared_name20bigger_auto_encoder/bigger_encoder/conv2d/kernel
�
Dbigger_auto_encoder/bigger_encoder/conv2d/kernel/Read/ReadVariableOpReadVariableOp0bigger_auto_encoder/bigger_encoder/conv2d/kernel*&
_output_shapes
:*
dtype0
�
.bigger_auto_encoder/bigger_encoder/conv2d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*?
shared_name0.bigger_auto_encoder/bigger_encoder/conv2d/bias
�
Bbigger_auto_encoder/bigger_encoder/conv2d/bias/Read/ReadVariableOpReadVariableOp.bigger_auto_encoder/bigger_encoder/conv2d/bias*
_output_shapes
:*
dtype0
�
2bigger_auto_encoder/bigger_encoder/conv2d_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *C
shared_name42bigger_auto_encoder/bigger_encoder/conv2d_1/kernel
�
Fbigger_auto_encoder/bigger_encoder/conv2d_1/kernel/Read/ReadVariableOpReadVariableOp2bigger_auto_encoder/bigger_encoder/conv2d_1/kernel*&
_output_shapes
: *
dtype0
�
0bigger_auto_encoder/bigger_encoder/conv2d_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *A
shared_name20bigger_auto_encoder/bigger_encoder/conv2d_1/bias
�
Dbigger_auto_encoder/bigger_encoder/conv2d_1/bias/Read/ReadVariableOpReadVariableOp0bigger_auto_encoder/bigger_encoder/conv2d_1/bias*
_output_shapes
: *
dtype0
�
2bigger_auto_encoder/bigger_encoder/conv2d_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*C
shared_name42bigger_auto_encoder/bigger_encoder/conv2d_2/kernel
�
Fbigger_auto_encoder/bigger_encoder/conv2d_2/kernel/Read/ReadVariableOpReadVariableOp2bigger_auto_encoder/bigger_encoder/conv2d_2/kernel*&
_output_shapes
: @*
dtype0
�
0bigger_auto_encoder/bigger_encoder/conv2d_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*A
shared_name20bigger_auto_encoder/bigger_encoder/conv2d_2/bias
�
Dbigger_auto_encoder/bigger_encoder/conv2d_2/bias/Read/ReadVariableOpReadVariableOp0bigger_auto_encoder/bigger_encoder/conv2d_2/bias*
_output_shapes
:@*
dtype0
�
2bigger_auto_encoder/bigger_encoder/conv2d_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*C
shared_name42bigger_auto_encoder/bigger_encoder/conv2d_3/kernel
�
Fbigger_auto_encoder/bigger_encoder/conv2d_3/kernel/Read/ReadVariableOpReadVariableOp2bigger_auto_encoder/bigger_encoder/conv2d_3/kernel*&
_output_shapes
:@@*
dtype0
�
0bigger_auto_encoder/bigger_encoder/conv2d_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*A
shared_name20bigger_auto_encoder/bigger_encoder/conv2d_3/bias
�
Dbigger_auto_encoder/bigger_encoder/conv2d_3/bias/Read/ReadVariableOpReadVariableOp0bigger_auto_encoder/bigger_encoder/conv2d_3/bias*
_output_shapes
:@*
dtype0
�
:bigger_auto_encoder/bigger_decoder/conv2d_transpose/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*K
shared_name<:bigger_auto_encoder/bigger_decoder/conv2d_transpose/kernel
�
Nbigger_auto_encoder/bigger_decoder/conv2d_transpose/kernel/Read/ReadVariableOpReadVariableOp:bigger_auto_encoder/bigger_decoder/conv2d_transpose/kernel*&
_output_shapes
:@@*
dtype0
�
8bigger_auto_encoder/bigger_decoder/conv2d_transpose/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*I
shared_name:8bigger_auto_encoder/bigger_decoder/conv2d_transpose/bias
�
Lbigger_auto_encoder/bigger_decoder/conv2d_transpose/bias/Read/ReadVariableOpReadVariableOp8bigger_auto_encoder/bigger_decoder/conv2d_transpose/bias*
_output_shapes
:@*
dtype0
�
<bigger_auto_encoder/bigger_decoder/conv2d_transpose_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*M
shared_name><bigger_auto_encoder/bigger_decoder/conv2d_transpose_1/kernel
�
Pbigger_auto_encoder/bigger_decoder/conv2d_transpose_1/kernel/Read/ReadVariableOpReadVariableOp<bigger_auto_encoder/bigger_decoder/conv2d_transpose_1/kernel*&
_output_shapes
:@@*
dtype0
�
:bigger_auto_encoder/bigger_decoder/conv2d_transpose_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*K
shared_name<:bigger_auto_encoder/bigger_decoder/conv2d_transpose_1/bias
�
Nbigger_auto_encoder/bigger_decoder/conv2d_transpose_1/bias/Read/ReadVariableOpReadVariableOp:bigger_auto_encoder/bigger_decoder/conv2d_transpose_1/bias*
_output_shapes
:@*
dtype0
�
<bigger_auto_encoder/bigger_decoder/conv2d_transpose_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*M
shared_name><bigger_auto_encoder/bigger_decoder/conv2d_transpose_2/kernel
�
Pbigger_auto_encoder/bigger_decoder/conv2d_transpose_2/kernel/Read/ReadVariableOpReadVariableOp<bigger_auto_encoder/bigger_decoder/conv2d_transpose_2/kernel*&
_output_shapes
: @*
dtype0
�
:bigger_auto_encoder/bigger_decoder/conv2d_transpose_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *K
shared_name<:bigger_auto_encoder/bigger_decoder/conv2d_transpose_2/bias
�
Nbigger_auto_encoder/bigger_decoder/conv2d_transpose_2/bias/Read/ReadVariableOpReadVariableOp:bigger_auto_encoder/bigger_decoder/conv2d_transpose_2/bias*
_output_shapes
: *
dtype0
�
<bigger_auto_encoder/bigger_decoder/conv2d_transpose_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *M
shared_name><bigger_auto_encoder/bigger_decoder/conv2d_transpose_3/kernel
�
Pbigger_auto_encoder/bigger_decoder/conv2d_transpose_3/kernel/Read/ReadVariableOpReadVariableOp<bigger_auto_encoder/bigger_decoder/conv2d_transpose_3/kernel*&
_output_shapes
: *
dtype0
�
:bigger_auto_encoder/bigger_decoder/conv2d_transpose_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*K
shared_name<:bigger_auto_encoder/bigger_decoder/conv2d_transpose_3/bias
�
Nbigger_auto_encoder/bigger_decoder/conv2d_transpose_3/bias/Read/ReadVariableOpReadVariableOp:bigger_auto_encoder/bigger_decoder/conv2d_transpose_3/bias*
_output_shapes
:*
dtype0
�
2bigger_auto_encoder/bigger_decoder/conv2d_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*C
shared_name42bigger_auto_encoder/bigger_decoder/conv2d_4/kernel
�
Fbigger_auto_encoder/bigger_decoder/conv2d_4/kernel/Read/ReadVariableOpReadVariableOp2bigger_auto_encoder/bigger_decoder/conv2d_4/kernel*&
_output_shapes
:*
dtype0
�
0bigger_auto_encoder/bigger_decoder/conv2d_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*A
shared_name20bigger_auto_encoder/bigger_decoder/conv2d_4/bias
�
Dbigger_auto_encoder/bigger_decoder/conv2d_4/bias/Read/ReadVariableOpReadVariableOp0bigger_auto_encoder/bigger_decoder/conv2d_4/bias*
_output_shapes
:*
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
�
7Adam/bigger_auto_encoder/bigger_encoder/conv2d/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*H
shared_name97Adam/bigger_auto_encoder/bigger_encoder/conv2d/kernel/m
�
KAdam/bigger_auto_encoder/bigger_encoder/conv2d/kernel/m/Read/ReadVariableOpReadVariableOp7Adam/bigger_auto_encoder/bigger_encoder/conv2d/kernel/m*&
_output_shapes
:*
dtype0
�
5Adam/bigger_auto_encoder/bigger_encoder/conv2d/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*F
shared_name75Adam/bigger_auto_encoder/bigger_encoder/conv2d/bias/m
�
IAdam/bigger_auto_encoder/bigger_encoder/conv2d/bias/m/Read/ReadVariableOpReadVariableOp5Adam/bigger_auto_encoder/bigger_encoder/conv2d/bias/m*
_output_shapes
:*
dtype0
�
9Adam/bigger_auto_encoder/bigger_encoder/conv2d_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *J
shared_name;9Adam/bigger_auto_encoder/bigger_encoder/conv2d_1/kernel/m
�
MAdam/bigger_auto_encoder/bigger_encoder/conv2d_1/kernel/m/Read/ReadVariableOpReadVariableOp9Adam/bigger_auto_encoder/bigger_encoder/conv2d_1/kernel/m*&
_output_shapes
: *
dtype0
�
7Adam/bigger_auto_encoder/bigger_encoder/conv2d_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *H
shared_name97Adam/bigger_auto_encoder/bigger_encoder/conv2d_1/bias/m
�
KAdam/bigger_auto_encoder/bigger_encoder/conv2d_1/bias/m/Read/ReadVariableOpReadVariableOp7Adam/bigger_auto_encoder/bigger_encoder/conv2d_1/bias/m*
_output_shapes
: *
dtype0
�
9Adam/bigger_auto_encoder/bigger_encoder/conv2d_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*J
shared_name;9Adam/bigger_auto_encoder/bigger_encoder/conv2d_2/kernel/m
�
MAdam/bigger_auto_encoder/bigger_encoder/conv2d_2/kernel/m/Read/ReadVariableOpReadVariableOp9Adam/bigger_auto_encoder/bigger_encoder/conv2d_2/kernel/m*&
_output_shapes
: @*
dtype0
�
7Adam/bigger_auto_encoder/bigger_encoder/conv2d_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*H
shared_name97Adam/bigger_auto_encoder/bigger_encoder/conv2d_2/bias/m
�
KAdam/bigger_auto_encoder/bigger_encoder/conv2d_2/bias/m/Read/ReadVariableOpReadVariableOp7Adam/bigger_auto_encoder/bigger_encoder/conv2d_2/bias/m*
_output_shapes
:@*
dtype0
�
9Adam/bigger_auto_encoder/bigger_encoder/conv2d_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*J
shared_name;9Adam/bigger_auto_encoder/bigger_encoder/conv2d_3/kernel/m
�
MAdam/bigger_auto_encoder/bigger_encoder/conv2d_3/kernel/m/Read/ReadVariableOpReadVariableOp9Adam/bigger_auto_encoder/bigger_encoder/conv2d_3/kernel/m*&
_output_shapes
:@@*
dtype0
�
7Adam/bigger_auto_encoder/bigger_encoder/conv2d_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*H
shared_name97Adam/bigger_auto_encoder/bigger_encoder/conv2d_3/bias/m
�
KAdam/bigger_auto_encoder/bigger_encoder/conv2d_3/bias/m/Read/ReadVariableOpReadVariableOp7Adam/bigger_auto_encoder/bigger_encoder/conv2d_3/bias/m*
_output_shapes
:@*
dtype0
�
AAdam/bigger_auto_encoder/bigger_decoder/conv2d_transpose/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*R
shared_nameCAAdam/bigger_auto_encoder/bigger_decoder/conv2d_transpose/kernel/m
�
UAdam/bigger_auto_encoder/bigger_decoder/conv2d_transpose/kernel/m/Read/ReadVariableOpReadVariableOpAAdam/bigger_auto_encoder/bigger_decoder/conv2d_transpose/kernel/m*&
_output_shapes
:@@*
dtype0
�
?Adam/bigger_auto_encoder/bigger_decoder/conv2d_transpose/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*P
shared_nameA?Adam/bigger_auto_encoder/bigger_decoder/conv2d_transpose/bias/m
�
SAdam/bigger_auto_encoder/bigger_decoder/conv2d_transpose/bias/m/Read/ReadVariableOpReadVariableOp?Adam/bigger_auto_encoder/bigger_decoder/conv2d_transpose/bias/m*
_output_shapes
:@*
dtype0
�
CAdam/bigger_auto_encoder/bigger_decoder/conv2d_transpose_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*T
shared_nameECAdam/bigger_auto_encoder/bigger_decoder/conv2d_transpose_1/kernel/m
�
WAdam/bigger_auto_encoder/bigger_decoder/conv2d_transpose_1/kernel/m/Read/ReadVariableOpReadVariableOpCAdam/bigger_auto_encoder/bigger_decoder/conv2d_transpose_1/kernel/m*&
_output_shapes
:@@*
dtype0
�
AAdam/bigger_auto_encoder/bigger_decoder/conv2d_transpose_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*R
shared_nameCAAdam/bigger_auto_encoder/bigger_decoder/conv2d_transpose_1/bias/m
�
UAdam/bigger_auto_encoder/bigger_decoder/conv2d_transpose_1/bias/m/Read/ReadVariableOpReadVariableOpAAdam/bigger_auto_encoder/bigger_decoder/conv2d_transpose_1/bias/m*
_output_shapes
:@*
dtype0
�
CAdam/bigger_auto_encoder/bigger_decoder/conv2d_transpose_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*T
shared_nameECAdam/bigger_auto_encoder/bigger_decoder/conv2d_transpose_2/kernel/m
�
WAdam/bigger_auto_encoder/bigger_decoder/conv2d_transpose_2/kernel/m/Read/ReadVariableOpReadVariableOpCAdam/bigger_auto_encoder/bigger_decoder/conv2d_transpose_2/kernel/m*&
_output_shapes
: @*
dtype0
�
AAdam/bigger_auto_encoder/bigger_decoder/conv2d_transpose_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *R
shared_nameCAAdam/bigger_auto_encoder/bigger_decoder/conv2d_transpose_2/bias/m
�
UAdam/bigger_auto_encoder/bigger_decoder/conv2d_transpose_2/bias/m/Read/ReadVariableOpReadVariableOpAAdam/bigger_auto_encoder/bigger_decoder/conv2d_transpose_2/bias/m*
_output_shapes
: *
dtype0
�
CAdam/bigger_auto_encoder/bigger_decoder/conv2d_transpose_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *T
shared_nameECAdam/bigger_auto_encoder/bigger_decoder/conv2d_transpose_3/kernel/m
�
WAdam/bigger_auto_encoder/bigger_decoder/conv2d_transpose_3/kernel/m/Read/ReadVariableOpReadVariableOpCAdam/bigger_auto_encoder/bigger_decoder/conv2d_transpose_3/kernel/m*&
_output_shapes
: *
dtype0
�
AAdam/bigger_auto_encoder/bigger_decoder/conv2d_transpose_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*R
shared_nameCAAdam/bigger_auto_encoder/bigger_decoder/conv2d_transpose_3/bias/m
�
UAdam/bigger_auto_encoder/bigger_decoder/conv2d_transpose_3/bias/m/Read/ReadVariableOpReadVariableOpAAdam/bigger_auto_encoder/bigger_decoder/conv2d_transpose_3/bias/m*
_output_shapes
:*
dtype0
�
9Adam/bigger_auto_encoder/bigger_decoder/conv2d_4/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*J
shared_name;9Adam/bigger_auto_encoder/bigger_decoder/conv2d_4/kernel/m
�
MAdam/bigger_auto_encoder/bigger_decoder/conv2d_4/kernel/m/Read/ReadVariableOpReadVariableOp9Adam/bigger_auto_encoder/bigger_decoder/conv2d_4/kernel/m*&
_output_shapes
:*
dtype0
�
7Adam/bigger_auto_encoder/bigger_decoder/conv2d_4/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*H
shared_name97Adam/bigger_auto_encoder/bigger_decoder/conv2d_4/bias/m
�
KAdam/bigger_auto_encoder/bigger_decoder/conv2d_4/bias/m/Read/ReadVariableOpReadVariableOp7Adam/bigger_auto_encoder/bigger_decoder/conv2d_4/bias/m*
_output_shapes
:*
dtype0
�
7Adam/bigger_auto_encoder/bigger_encoder/conv2d/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*H
shared_name97Adam/bigger_auto_encoder/bigger_encoder/conv2d/kernel/v
�
KAdam/bigger_auto_encoder/bigger_encoder/conv2d/kernel/v/Read/ReadVariableOpReadVariableOp7Adam/bigger_auto_encoder/bigger_encoder/conv2d/kernel/v*&
_output_shapes
:*
dtype0
�
5Adam/bigger_auto_encoder/bigger_encoder/conv2d/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*F
shared_name75Adam/bigger_auto_encoder/bigger_encoder/conv2d/bias/v
�
IAdam/bigger_auto_encoder/bigger_encoder/conv2d/bias/v/Read/ReadVariableOpReadVariableOp5Adam/bigger_auto_encoder/bigger_encoder/conv2d/bias/v*
_output_shapes
:*
dtype0
�
9Adam/bigger_auto_encoder/bigger_encoder/conv2d_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *J
shared_name;9Adam/bigger_auto_encoder/bigger_encoder/conv2d_1/kernel/v
�
MAdam/bigger_auto_encoder/bigger_encoder/conv2d_1/kernel/v/Read/ReadVariableOpReadVariableOp9Adam/bigger_auto_encoder/bigger_encoder/conv2d_1/kernel/v*&
_output_shapes
: *
dtype0
�
7Adam/bigger_auto_encoder/bigger_encoder/conv2d_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *H
shared_name97Adam/bigger_auto_encoder/bigger_encoder/conv2d_1/bias/v
�
KAdam/bigger_auto_encoder/bigger_encoder/conv2d_1/bias/v/Read/ReadVariableOpReadVariableOp7Adam/bigger_auto_encoder/bigger_encoder/conv2d_1/bias/v*
_output_shapes
: *
dtype0
�
9Adam/bigger_auto_encoder/bigger_encoder/conv2d_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*J
shared_name;9Adam/bigger_auto_encoder/bigger_encoder/conv2d_2/kernel/v
�
MAdam/bigger_auto_encoder/bigger_encoder/conv2d_2/kernel/v/Read/ReadVariableOpReadVariableOp9Adam/bigger_auto_encoder/bigger_encoder/conv2d_2/kernel/v*&
_output_shapes
: @*
dtype0
�
7Adam/bigger_auto_encoder/bigger_encoder/conv2d_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*H
shared_name97Adam/bigger_auto_encoder/bigger_encoder/conv2d_2/bias/v
�
KAdam/bigger_auto_encoder/bigger_encoder/conv2d_2/bias/v/Read/ReadVariableOpReadVariableOp7Adam/bigger_auto_encoder/bigger_encoder/conv2d_2/bias/v*
_output_shapes
:@*
dtype0
�
9Adam/bigger_auto_encoder/bigger_encoder/conv2d_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*J
shared_name;9Adam/bigger_auto_encoder/bigger_encoder/conv2d_3/kernel/v
�
MAdam/bigger_auto_encoder/bigger_encoder/conv2d_3/kernel/v/Read/ReadVariableOpReadVariableOp9Adam/bigger_auto_encoder/bigger_encoder/conv2d_3/kernel/v*&
_output_shapes
:@@*
dtype0
�
7Adam/bigger_auto_encoder/bigger_encoder/conv2d_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*H
shared_name97Adam/bigger_auto_encoder/bigger_encoder/conv2d_3/bias/v
�
KAdam/bigger_auto_encoder/bigger_encoder/conv2d_3/bias/v/Read/ReadVariableOpReadVariableOp7Adam/bigger_auto_encoder/bigger_encoder/conv2d_3/bias/v*
_output_shapes
:@*
dtype0
�
AAdam/bigger_auto_encoder/bigger_decoder/conv2d_transpose/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*R
shared_nameCAAdam/bigger_auto_encoder/bigger_decoder/conv2d_transpose/kernel/v
�
UAdam/bigger_auto_encoder/bigger_decoder/conv2d_transpose/kernel/v/Read/ReadVariableOpReadVariableOpAAdam/bigger_auto_encoder/bigger_decoder/conv2d_transpose/kernel/v*&
_output_shapes
:@@*
dtype0
�
?Adam/bigger_auto_encoder/bigger_decoder/conv2d_transpose/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*P
shared_nameA?Adam/bigger_auto_encoder/bigger_decoder/conv2d_transpose/bias/v
�
SAdam/bigger_auto_encoder/bigger_decoder/conv2d_transpose/bias/v/Read/ReadVariableOpReadVariableOp?Adam/bigger_auto_encoder/bigger_decoder/conv2d_transpose/bias/v*
_output_shapes
:@*
dtype0
�
CAdam/bigger_auto_encoder/bigger_decoder/conv2d_transpose_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*T
shared_nameECAdam/bigger_auto_encoder/bigger_decoder/conv2d_transpose_1/kernel/v
�
WAdam/bigger_auto_encoder/bigger_decoder/conv2d_transpose_1/kernel/v/Read/ReadVariableOpReadVariableOpCAdam/bigger_auto_encoder/bigger_decoder/conv2d_transpose_1/kernel/v*&
_output_shapes
:@@*
dtype0
�
AAdam/bigger_auto_encoder/bigger_decoder/conv2d_transpose_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*R
shared_nameCAAdam/bigger_auto_encoder/bigger_decoder/conv2d_transpose_1/bias/v
�
UAdam/bigger_auto_encoder/bigger_decoder/conv2d_transpose_1/bias/v/Read/ReadVariableOpReadVariableOpAAdam/bigger_auto_encoder/bigger_decoder/conv2d_transpose_1/bias/v*
_output_shapes
:@*
dtype0
�
CAdam/bigger_auto_encoder/bigger_decoder/conv2d_transpose_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*T
shared_nameECAdam/bigger_auto_encoder/bigger_decoder/conv2d_transpose_2/kernel/v
�
WAdam/bigger_auto_encoder/bigger_decoder/conv2d_transpose_2/kernel/v/Read/ReadVariableOpReadVariableOpCAdam/bigger_auto_encoder/bigger_decoder/conv2d_transpose_2/kernel/v*&
_output_shapes
: @*
dtype0
�
AAdam/bigger_auto_encoder/bigger_decoder/conv2d_transpose_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *R
shared_nameCAAdam/bigger_auto_encoder/bigger_decoder/conv2d_transpose_2/bias/v
�
UAdam/bigger_auto_encoder/bigger_decoder/conv2d_transpose_2/bias/v/Read/ReadVariableOpReadVariableOpAAdam/bigger_auto_encoder/bigger_decoder/conv2d_transpose_2/bias/v*
_output_shapes
: *
dtype0
�
CAdam/bigger_auto_encoder/bigger_decoder/conv2d_transpose_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *T
shared_nameECAdam/bigger_auto_encoder/bigger_decoder/conv2d_transpose_3/kernel/v
�
WAdam/bigger_auto_encoder/bigger_decoder/conv2d_transpose_3/kernel/v/Read/ReadVariableOpReadVariableOpCAdam/bigger_auto_encoder/bigger_decoder/conv2d_transpose_3/kernel/v*&
_output_shapes
: *
dtype0
�
AAdam/bigger_auto_encoder/bigger_decoder/conv2d_transpose_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*R
shared_nameCAAdam/bigger_auto_encoder/bigger_decoder/conv2d_transpose_3/bias/v
�
UAdam/bigger_auto_encoder/bigger_decoder/conv2d_transpose_3/bias/v/Read/ReadVariableOpReadVariableOpAAdam/bigger_auto_encoder/bigger_decoder/conv2d_transpose_3/bias/v*
_output_shapes
:*
dtype0
�
9Adam/bigger_auto_encoder/bigger_decoder/conv2d_4/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*J
shared_name;9Adam/bigger_auto_encoder/bigger_decoder/conv2d_4/kernel/v
�
MAdam/bigger_auto_encoder/bigger_decoder/conv2d_4/kernel/v/Read/ReadVariableOpReadVariableOp9Adam/bigger_auto_encoder/bigger_decoder/conv2d_4/kernel/v*&
_output_shapes
:*
dtype0
�
7Adam/bigger_auto_encoder/bigger_decoder/conv2d_4/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*H
shared_name97Adam/bigger_auto_encoder/bigger_decoder/conv2d_4/bias/v
�
KAdam/bigger_auto_encoder/bigger_decoder/conv2d_4/bias/v/Read/ReadVariableOpReadVariableOp7Adam/bigger_auto_encoder/bigger_decoder/conv2d_4/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
�p
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�p
value�pB�p B�p
�
encoder
decoder
	optimizer
regularization_losses
trainable_variables
	variables
	keras_api

signatures
�

	conv_1


conv_2

conv_3

conv_4
regularization_losses
trainable_variables
	variables
	keras_api
�

conv_1

conv_2

conv_3

conv_4
upsampling_4
conv_out
regularization_losses
trainable_variables
	variables
	keras_api
�
iter

beta_1

beta_2
	decay
learning_rate m�!m�"m�#m�$m�%m�&m�'m�(m�)m�*m�+m�,m�-m�.m�/m�0m�1m� v�!v�"v�#v�$v�%v�&v�'v�(v�)v�*v�+v�,v�-v�.v�/v�0v�1v�
 
�
 0
!1
"2
#3
$4
%5
&6
'7
(8
)9
*10
+11
,12
-13
.14
/15
016
117
�
 0
!1
"2
#3
$4
%5
&6
'7
(8
)9
*10
+11
,12
-13
.14
/15
016
117
�
regularization_losses

2layers
3layer_metrics
4metrics
trainable_variables
5layer_regularization_losses
6non_trainable_variables
	variables
 
h

 kernel
!bias
7regularization_losses
8trainable_variables
9	variables
:	keras_api
h

"kernel
#bias
;regularization_losses
<trainable_variables
=	variables
>	keras_api
h

$kernel
%bias
?regularization_losses
@trainable_variables
A	variables
B	keras_api
h

&kernel
'bias
Cregularization_losses
Dtrainable_variables
E	variables
F	keras_api
 
8
 0
!1
"2
#3
$4
%5
&6
'7
8
 0
!1
"2
#3
$4
%5
&6
'7
�
regularization_losses

Glayers
Hlayer_metrics
Imetrics
trainable_variables
Jlayer_regularization_losses
Knon_trainable_variables
	variables
h

(kernel
)bias
Lregularization_losses
Mtrainable_variables
N	variables
O	keras_api
h

*kernel
+bias
Pregularization_losses
Qtrainable_variables
R	variables
S	keras_api
h

,kernel
-bias
Tregularization_losses
Utrainable_variables
V	variables
W	keras_api
h

.kernel
/bias
Xregularization_losses
Ytrainable_variables
Z	variables
[	keras_api
R
\regularization_losses
]trainable_variables
^	variables
_	keras_api
h

0kernel
1bias
`regularization_losses
atrainable_variables
b	variables
c	keras_api
 
F
(0
)1
*2
+3
,4
-5
.6
/7
08
19
F
(0
)1
*2
+3
,4
-5
.6
/7
08
19
�
regularization_losses

dlayers
elayer_metrics
fmetrics
trainable_variables
glayer_regularization_losses
hnon_trainable_variables
	variables
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUE0bigger_auto_encoder/bigger_encoder/conv2d/kernel0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUE.bigger_auto_encoder/bigger_encoder/conv2d/bias0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUE2bigger_auto_encoder/bigger_encoder/conv2d_1/kernel0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUE0bigger_auto_encoder/bigger_encoder/conv2d_1/bias0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUE2bigger_auto_encoder/bigger_encoder/conv2d_2/kernel0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUE0bigger_auto_encoder/bigger_encoder/conv2d_2/bias0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUE2bigger_auto_encoder/bigger_encoder/conv2d_3/kernel0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUE0bigger_auto_encoder/bigger_encoder/conv2d_3/bias0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUE
�~
VARIABLE_VALUE:bigger_auto_encoder/bigger_decoder/conv2d_transpose/kernel0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUE8bigger_auto_encoder/bigger_decoder/conv2d_transpose/bias0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE<bigger_auto_encoder/bigger_decoder/conv2d_transpose_1/kernel1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUE
�
VARIABLE_VALUE:bigger_auto_encoder/bigger_decoder/conv2d_transpose_1/bias1trainable_variables/11/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE<bigger_auto_encoder/bigger_decoder/conv2d_transpose_2/kernel1trainable_variables/12/.ATTRIBUTES/VARIABLE_VALUE
�
VARIABLE_VALUE:bigger_auto_encoder/bigger_decoder/conv2d_transpose_2/bias1trainable_variables/13/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE<bigger_auto_encoder/bigger_decoder/conv2d_transpose_3/kernel1trainable_variables/14/.ATTRIBUTES/VARIABLE_VALUE
�
VARIABLE_VALUE:bigger_auto_encoder/bigger_decoder/conv2d_transpose_3/bias1trainable_variables/15/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUE2bigger_auto_encoder/bigger_decoder/conv2d_4/kernel1trainable_variables/16/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUE0bigger_auto_encoder/bigger_decoder/conv2d_4/bias1trainable_variables/17/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

i0
j1
 
 
 

 0
!1

 0
!1
�
7regularization_losses
knon_trainable_variables

llayers
mmetrics
8trainable_variables
nlayer_regularization_losses
olayer_metrics
9	variables
 

"0
#1

"0
#1
�
;regularization_losses
pnon_trainable_variables

qlayers
rmetrics
<trainable_variables
slayer_regularization_losses
tlayer_metrics
=	variables
 

$0
%1

$0
%1
�
?regularization_losses
unon_trainable_variables

vlayers
wmetrics
@trainable_variables
xlayer_regularization_losses
ylayer_metrics
A	variables
 

&0
'1

&0
'1
�
Cregularization_losses
znon_trainable_variables

{layers
|metrics
Dtrainable_variables
}layer_regularization_losses
~layer_metrics
E	variables

	0

1
2
3
 
 
 
 
 

(0
)1

(0
)1
�
Lregularization_losses
non_trainable_variables
�layers
�metrics
Mtrainable_variables
 �layer_regularization_losses
�layer_metrics
N	variables
 

*0
+1

*0
+1
�
Pregularization_losses
�non_trainable_variables
�layers
�metrics
Qtrainable_variables
 �layer_regularization_losses
�layer_metrics
R	variables
 

,0
-1

,0
-1
�
Tregularization_losses
�non_trainable_variables
�layers
�metrics
Utrainable_variables
 �layer_regularization_losses
�layer_metrics
V	variables
 

.0
/1

.0
/1
�
Xregularization_losses
�non_trainable_variables
�layers
�metrics
Ytrainable_variables
 �layer_regularization_losses
�layer_metrics
Z	variables
 
 
 
�
\regularization_losses
�non_trainable_variables
�layers
�metrics
]trainable_variables
 �layer_regularization_losses
�layer_metrics
^	variables
 

00
11

00
11
�
`regularization_losses
�non_trainable_variables
�layers
�metrics
atrainable_variables
 �layer_regularization_losses
�layer_metrics
b	variables
*
0
1
2
3
4
5
 
 
 
 
8

�total

�count
�	variables
�	keras_api
I

�total

�count
�
_fn_kwargs
�	variables
�	keras_api
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

�0
�1

�	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

�0
�1

�	variables
��
VARIABLE_VALUE7Adam/bigger_auto_encoder/bigger_encoder/conv2d/kernel/mLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE5Adam/bigger_auto_encoder/bigger_encoder/conv2d/bias/mLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE9Adam/bigger_auto_encoder/bigger_encoder/conv2d_1/kernel/mLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE7Adam/bigger_auto_encoder/bigger_encoder/conv2d_1/bias/mLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE9Adam/bigger_auto_encoder/bigger_encoder/conv2d_2/kernel/mLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE7Adam/bigger_auto_encoder/bigger_encoder/conv2d_2/bias/mLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE9Adam/bigger_auto_encoder/bigger_encoder/conv2d_3/kernel/mLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE7Adam/bigger_auto_encoder/bigger_encoder/conv2d_3/bias/mLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUEAAdam/bigger_auto_encoder/bigger_decoder/conv2d_transpose/kernel/mLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE?Adam/bigger_auto_encoder/bigger_decoder/conv2d_transpose/bias/mLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUECAdam/bigger_auto_encoder/bigger_decoder/conv2d_transpose_1/kernel/mMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUEAAdam/bigger_auto_encoder/bigger_decoder/conv2d_transpose_1/bias/mMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUECAdam/bigger_auto_encoder/bigger_decoder/conv2d_transpose_2/kernel/mMtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUEAAdam/bigger_auto_encoder/bigger_decoder/conv2d_transpose_2/bias/mMtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUECAdam/bigger_auto_encoder/bigger_decoder/conv2d_transpose_3/kernel/mMtrainable_variables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUEAAdam/bigger_auto_encoder/bigger_decoder/conv2d_transpose_3/bias/mMtrainable_variables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE9Adam/bigger_auto_encoder/bigger_decoder/conv2d_4/kernel/mMtrainable_variables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE7Adam/bigger_auto_encoder/bigger_decoder/conv2d_4/bias/mMtrainable_variables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE7Adam/bigger_auto_encoder/bigger_encoder/conv2d/kernel/vLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE5Adam/bigger_auto_encoder/bigger_encoder/conv2d/bias/vLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE9Adam/bigger_auto_encoder/bigger_encoder/conv2d_1/kernel/vLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE7Adam/bigger_auto_encoder/bigger_encoder/conv2d_1/bias/vLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE9Adam/bigger_auto_encoder/bigger_encoder/conv2d_2/kernel/vLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE7Adam/bigger_auto_encoder/bigger_encoder/conv2d_2/bias/vLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE9Adam/bigger_auto_encoder/bigger_encoder/conv2d_3/kernel/vLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE7Adam/bigger_auto_encoder/bigger_encoder/conv2d_3/bias/vLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUEAAdam/bigger_auto_encoder/bigger_decoder/conv2d_transpose/kernel/vLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE?Adam/bigger_auto_encoder/bigger_decoder/conv2d_transpose/bias/vLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUECAdam/bigger_auto_encoder/bigger_decoder/conv2d_transpose_1/kernel/vMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUEAAdam/bigger_auto_encoder/bigger_decoder/conv2d_transpose_1/bias/vMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUECAdam/bigger_auto_encoder/bigger_decoder/conv2d_transpose_2/kernel/vMtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUEAAdam/bigger_auto_encoder/bigger_decoder/conv2d_transpose_2/bias/vMtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUECAdam/bigger_auto_encoder/bigger_decoder/conv2d_transpose_3/kernel/vMtrainable_variables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUEAAdam/bigger_auto_encoder/bigger_decoder/conv2d_transpose_3/bias/vMtrainable_variables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE9Adam/bigger_auto_encoder/bigger_decoder/conv2d_4/kernel/vMtrainable_variables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE7Adam/bigger_auto_encoder/bigger_decoder/conv2d_4/bias/vMtrainable_variables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
�
serving_default_input_1Placeholder*1
_output_shapes
:�����������*
dtype0*&
shape:�����������
�

StatefulPartitionedCallStatefulPartitionedCallserving_default_input_10bigger_auto_encoder/bigger_encoder/conv2d/kernel.bigger_auto_encoder/bigger_encoder/conv2d/bias2bigger_auto_encoder/bigger_encoder/conv2d_1/kernel0bigger_auto_encoder/bigger_encoder/conv2d_1/bias2bigger_auto_encoder/bigger_encoder/conv2d_2/kernel0bigger_auto_encoder/bigger_encoder/conv2d_2/bias2bigger_auto_encoder/bigger_encoder/conv2d_3/kernel0bigger_auto_encoder/bigger_encoder/conv2d_3/bias:bigger_auto_encoder/bigger_decoder/conv2d_transpose/kernel8bigger_auto_encoder/bigger_decoder/conv2d_transpose/bias<bigger_auto_encoder/bigger_decoder/conv2d_transpose_1/kernel:bigger_auto_encoder/bigger_decoder/conv2d_transpose_1/bias<bigger_auto_encoder/bigger_decoder/conv2d_transpose_2/kernel:bigger_auto_encoder/bigger_decoder/conv2d_transpose_2/bias<bigger_auto_encoder/bigger_decoder/conv2d_transpose_3/kernel:bigger_auto_encoder/bigger_decoder/conv2d_transpose_3/bias2bigger_auto_encoder/bigger_decoder/conv2d_4/kernel0bigger_auto_encoder/bigger_decoder/conv2d_4/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������*4
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8� *-
f(R&
$__inference_signature_wrapper_144275
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�&
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenameAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOpDbigger_auto_encoder/bigger_encoder/conv2d/kernel/Read/ReadVariableOpBbigger_auto_encoder/bigger_encoder/conv2d/bias/Read/ReadVariableOpFbigger_auto_encoder/bigger_encoder/conv2d_1/kernel/Read/ReadVariableOpDbigger_auto_encoder/bigger_encoder/conv2d_1/bias/Read/ReadVariableOpFbigger_auto_encoder/bigger_encoder/conv2d_2/kernel/Read/ReadVariableOpDbigger_auto_encoder/bigger_encoder/conv2d_2/bias/Read/ReadVariableOpFbigger_auto_encoder/bigger_encoder/conv2d_3/kernel/Read/ReadVariableOpDbigger_auto_encoder/bigger_encoder/conv2d_3/bias/Read/ReadVariableOpNbigger_auto_encoder/bigger_decoder/conv2d_transpose/kernel/Read/ReadVariableOpLbigger_auto_encoder/bigger_decoder/conv2d_transpose/bias/Read/ReadVariableOpPbigger_auto_encoder/bigger_decoder/conv2d_transpose_1/kernel/Read/ReadVariableOpNbigger_auto_encoder/bigger_decoder/conv2d_transpose_1/bias/Read/ReadVariableOpPbigger_auto_encoder/bigger_decoder/conv2d_transpose_2/kernel/Read/ReadVariableOpNbigger_auto_encoder/bigger_decoder/conv2d_transpose_2/bias/Read/ReadVariableOpPbigger_auto_encoder/bigger_decoder/conv2d_transpose_3/kernel/Read/ReadVariableOpNbigger_auto_encoder/bigger_decoder/conv2d_transpose_3/bias/Read/ReadVariableOpFbigger_auto_encoder/bigger_decoder/conv2d_4/kernel/Read/ReadVariableOpDbigger_auto_encoder/bigger_decoder/conv2d_4/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOpKAdam/bigger_auto_encoder/bigger_encoder/conv2d/kernel/m/Read/ReadVariableOpIAdam/bigger_auto_encoder/bigger_encoder/conv2d/bias/m/Read/ReadVariableOpMAdam/bigger_auto_encoder/bigger_encoder/conv2d_1/kernel/m/Read/ReadVariableOpKAdam/bigger_auto_encoder/bigger_encoder/conv2d_1/bias/m/Read/ReadVariableOpMAdam/bigger_auto_encoder/bigger_encoder/conv2d_2/kernel/m/Read/ReadVariableOpKAdam/bigger_auto_encoder/bigger_encoder/conv2d_2/bias/m/Read/ReadVariableOpMAdam/bigger_auto_encoder/bigger_encoder/conv2d_3/kernel/m/Read/ReadVariableOpKAdam/bigger_auto_encoder/bigger_encoder/conv2d_3/bias/m/Read/ReadVariableOpUAdam/bigger_auto_encoder/bigger_decoder/conv2d_transpose/kernel/m/Read/ReadVariableOpSAdam/bigger_auto_encoder/bigger_decoder/conv2d_transpose/bias/m/Read/ReadVariableOpWAdam/bigger_auto_encoder/bigger_decoder/conv2d_transpose_1/kernel/m/Read/ReadVariableOpUAdam/bigger_auto_encoder/bigger_decoder/conv2d_transpose_1/bias/m/Read/ReadVariableOpWAdam/bigger_auto_encoder/bigger_decoder/conv2d_transpose_2/kernel/m/Read/ReadVariableOpUAdam/bigger_auto_encoder/bigger_decoder/conv2d_transpose_2/bias/m/Read/ReadVariableOpWAdam/bigger_auto_encoder/bigger_decoder/conv2d_transpose_3/kernel/m/Read/ReadVariableOpUAdam/bigger_auto_encoder/bigger_decoder/conv2d_transpose_3/bias/m/Read/ReadVariableOpMAdam/bigger_auto_encoder/bigger_decoder/conv2d_4/kernel/m/Read/ReadVariableOpKAdam/bigger_auto_encoder/bigger_decoder/conv2d_4/bias/m/Read/ReadVariableOpKAdam/bigger_auto_encoder/bigger_encoder/conv2d/kernel/v/Read/ReadVariableOpIAdam/bigger_auto_encoder/bigger_encoder/conv2d/bias/v/Read/ReadVariableOpMAdam/bigger_auto_encoder/bigger_encoder/conv2d_1/kernel/v/Read/ReadVariableOpKAdam/bigger_auto_encoder/bigger_encoder/conv2d_1/bias/v/Read/ReadVariableOpMAdam/bigger_auto_encoder/bigger_encoder/conv2d_2/kernel/v/Read/ReadVariableOpKAdam/bigger_auto_encoder/bigger_encoder/conv2d_2/bias/v/Read/ReadVariableOpMAdam/bigger_auto_encoder/bigger_encoder/conv2d_3/kernel/v/Read/ReadVariableOpKAdam/bigger_auto_encoder/bigger_encoder/conv2d_3/bias/v/Read/ReadVariableOpUAdam/bigger_auto_encoder/bigger_decoder/conv2d_transpose/kernel/v/Read/ReadVariableOpSAdam/bigger_auto_encoder/bigger_decoder/conv2d_transpose/bias/v/Read/ReadVariableOpWAdam/bigger_auto_encoder/bigger_decoder/conv2d_transpose_1/kernel/v/Read/ReadVariableOpUAdam/bigger_auto_encoder/bigger_decoder/conv2d_transpose_1/bias/v/Read/ReadVariableOpWAdam/bigger_auto_encoder/bigger_decoder/conv2d_transpose_2/kernel/v/Read/ReadVariableOpUAdam/bigger_auto_encoder/bigger_decoder/conv2d_transpose_2/bias/v/Read/ReadVariableOpWAdam/bigger_auto_encoder/bigger_decoder/conv2d_transpose_3/kernel/v/Read/ReadVariableOpUAdam/bigger_auto_encoder/bigger_decoder/conv2d_transpose_3/bias/v/Read/ReadVariableOpMAdam/bigger_auto_encoder/bigger_decoder/conv2d_4/kernel/v/Read/ReadVariableOpKAdam/bigger_auto_encoder/bigger_decoder/conv2d_4/bias/v/Read/ReadVariableOpConst*L
TinE
C2A	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *(
f#R!
__inference__traced_save_144587
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_rate0bigger_auto_encoder/bigger_encoder/conv2d/kernel.bigger_auto_encoder/bigger_encoder/conv2d/bias2bigger_auto_encoder/bigger_encoder/conv2d_1/kernel0bigger_auto_encoder/bigger_encoder/conv2d_1/bias2bigger_auto_encoder/bigger_encoder/conv2d_2/kernel0bigger_auto_encoder/bigger_encoder/conv2d_2/bias2bigger_auto_encoder/bigger_encoder/conv2d_3/kernel0bigger_auto_encoder/bigger_encoder/conv2d_3/bias:bigger_auto_encoder/bigger_decoder/conv2d_transpose/kernel8bigger_auto_encoder/bigger_decoder/conv2d_transpose/bias<bigger_auto_encoder/bigger_decoder/conv2d_transpose_1/kernel:bigger_auto_encoder/bigger_decoder/conv2d_transpose_1/bias<bigger_auto_encoder/bigger_decoder/conv2d_transpose_2/kernel:bigger_auto_encoder/bigger_decoder/conv2d_transpose_2/bias<bigger_auto_encoder/bigger_decoder/conv2d_transpose_3/kernel:bigger_auto_encoder/bigger_decoder/conv2d_transpose_3/bias2bigger_auto_encoder/bigger_decoder/conv2d_4/kernel0bigger_auto_encoder/bigger_decoder/conv2d_4/biastotalcounttotal_1count_17Adam/bigger_auto_encoder/bigger_encoder/conv2d/kernel/m5Adam/bigger_auto_encoder/bigger_encoder/conv2d/bias/m9Adam/bigger_auto_encoder/bigger_encoder/conv2d_1/kernel/m7Adam/bigger_auto_encoder/bigger_encoder/conv2d_1/bias/m9Adam/bigger_auto_encoder/bigger_encoder/conv2d_2/kernel/m7Adam/bigger_auto_encoder/bigger_encoder/conv2d_2/bias/m9Adam/bigger_auto_encoder/bigger_encoder/conv2d_3/kernel/m7Adam/bigger_auto_encoder/bigger_encoder/conv2d_3/bias/mAAdam/bigger_auto_encoder/bigger_decoder/conv2d_transpose/kernel/m?Adam/bigger_auto_encoder/bigger_decoder/conv2d_transpose/bias/mCAdam/bigger_auto_encoder/bigger_decoder/conv2d_transpose_1/kernel/mAAdam/bigger_auto_encoder/bigger_decoder/conv2d_transpose_1/bias/mCAdam/bigger_auto_encoder/bigger_decoder/conv2d_transpose_2/kernel/mAAdam/bigger_auto_encoder/bigger_decoder/conv2d_transpose_2/bias/mCAdam/bigger_auto_encoder/bigger_decoder/conv2d_transpose_3/kernel/mAAdam/bigger_auto_encoder/bigger_decoder/conv2d_transpose_3/bias/m9Adam/bigger_auto_encoder/bigger_decoder/conv2d_4/kernel/m7Adam/bigger_auto_encoder/bigger_decoder/conv2d_4/bias/m7Adam/bigger_auto_encoder/bigger_encoder/conv2d/kernel/v5Adam/bigger_auto_encoder/bigger_encoder/conv2d/bias/v9Adam/bigger_auto_encoder/bigger_encoder/conv2d_1/kernel/v7Adam/bigger_auto_encoder/bigger_encoder/conv2d_1/bias/v9Adam/bigger_auto_encoder/bigger_encoder/conv2d_2/kernel/v7Adam/bigger_auto_encoder/bigger_encoder/conv2d_2/bias/v9Adam/bigger_auto_encoder/bigger_encoder/conv2d_3/kernel/v7Adam/bigger_auto_encoder/bigger_encoder/conv2d_3/bias/vAAdam/bigger_auto_encoder/bigger_decoder/conv2d_transpose/kernel/v?Adam/bigger_auto_encoder/bigger_decoder/conv2d_transpose/bias/vCAdam/bigger_auto_encoder/bigger_decoder/conv2d_transpose_1/kernel/vAAdam/bigger_auto_encoder/bigger_decoder/conv2d_transpose_1/bias/vCAdam/bigger_auto_encoder/bigger_decoder/conv2d_transpose_2/kernel/vAAdam/bigger_auto_encoder/bigger_decoder/conv2d_transpose_2/bias/vCAdam/bigger_auto_encoder/bigger_decoder/conv2d_transpose_3/kernel/vAAdam/bigger_auto_encoder/bigger_decoder/conv2d_transpose_3/bias/v9Adam/bigger_auto_encoder/bigger_decoder/conv2d_4/kernel/v7Adam/bigger_auto_encoder/bigger_decoder/conv2d_4/bias/v*K
TinD
B2@*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *+
f&R$
"__inference__traced_restore_144786��	
�
|
'__inference_conv2d_layer_call_fn_144295

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *K
fFRD
B__inference_conv2d_layer_call_and_return_conditional_losses_1437412
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:�����������2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:�����������::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�#
�
J__inference_bigger_decoder_layer_call_and_return_conditional_losses_144113
input_1
conv2d_transpose_144064
conv2d_transpose_144066
conv2d_transpose_1_144069
conv2d_transpose_1_144071
conv2d_transpose_2_144074
conv2d_transpose_2_144076
conv2d_transpose_3_144079
conv2d_transpose_3_144081
conv2d_4_144107
conv2d_4_144109
identity�� conv2d_4/StatefulPartitionedCall�(conv2d_transpose/StatefulPartitionedCall�*conv2d_transpose_1/StatefulPartitionedCall�*conv2d_transpose_2/StatefulPartitionedCall�*conv2d_transpose_3/StatefulPartitionedCall�
(conv2d_transpose/StatefulPartitionedCallStatefulPartitionedCallinput_1conv2d_transpose_144064conv2d_transpose_144066*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *U
fPRN
L__inference_conv2d_transpose_layer_call_and_return_conditional_losses_1438962*
(conv2d_transpose/StatefulPartitionedCall�
*conv2d_transpose_1/StatefulPartitionedCallStatefulPartitionedCall1conv2d_transpose/StatefulPartitionedCall:output:0conv2d_transpose_1_144069conv2d_transpose_1_144071*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *W
fRRP
N__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_1439412,
*conv2d_transpose_1/StatefulPartitionedCall�
*conv2d_transpose_2/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_1/StatefulPartitionedCall:output:0conv2d_transpose_2_144074conv2d_transpose_2_144076*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+��������������������������� *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *W
fRRP
N__inference_conv2d_transpose_2_layer_call_and_return_conditional_losses_1439862,
*conv2d_transpose_2/StatefulPartitionedCall�
*conv2d_transpose_3/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_2/StatefulPartitionedCall:output:0conv2d_transpose_3_144079conv2d_transpose_3_144081*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *W
fRRP
N__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_1440312,
*conv2d_transpose_3/StatefulPartitionedCall�
up_sampling2d/PartitionedCallPartitionedCall3conv2d_transpose_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *R
fMRK
I__inference_up_sampling2d_layer_call_and_return_conditional_losses_1440542
up_sampling2d/PartitionedCall�
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCall&up_sampling2d/PartitionedCall:output:0conv2d_4_144107conv2d_4_144109*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *M
fHRF
D__inference_conv2d_4_layer_call_and_return_conditional_losses_1440962"
 conv2d_4/StatefulPartitionedCall�
IdentityIdentity)conv2d_4/StatefulPartitionedCall:output:0!^conv2d_4/StatefulPartitionedCall)^conv2d_transpose/StatefulPartitionedCall+^conv2d_transpose_1/StatefulPartitionedCall+^conv2d_transpose_2/StatefulPartitionedCall+^conv2d_transpose_3/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*X
_input_shapesG
E:�����������@::::::::::2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall2T
(conv2d_transpose/StatefulPartitionedCall(conv2d_transpose/StatefulPartitionedCall2X
*conv2d_transpose_1/StatefulPartitionedCall*conv2d_transpose_1/StatefulPartitionedCall2X
*conv2d_transpose_2/StatefulPartitionedCall*conv2d_transpose_2/StatefulPartitionedCall2X
*conv2d_transpose_3/StatefulPartitionedCall*conv2d_transpose_3/StatefulPartitionedCall:Z V
1
_output_shapes
:�����������@
!
_user_specified_name	input_1
�

�
D__inference_conv2d_3_layer_call_and_return_conditional_losses_143822

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������@*
paddingSAME*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������@2	
BiasAddb
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:�����������@2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:�����������@2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:�����������@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:�����������@
 
_user_specified_nameinputs
�
�
D__inference_conv2d_4_layer_call_and_return_conditional_losses_144096

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+���������������������������*
paddingSAME*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+���������������������������2	
BiasAddr
TanhTanhBiasAdd:output:0*
T0*A
_output_shapes/
-:+���������������������������2
Tanh�
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+���������������������������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�

�
$__inference_signature_wrapper_144275
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������*4
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8� **
f%R#
!__inference__wrapped_model_1437262
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:�����������2

Identity"
identityIdentity:output:0*x
_input_shapesg
e:�����������::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:�����������
!
_user_specified_name	input_1
�

�
B__inference_conv2d_layer_call_and_return_conditional_losses_144286

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������*
paddingSAME*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������2	
BiasAddb
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:�����������2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:�����������2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:�����������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�
~
)__inference_conv2d_2_layer_call_fn_144335

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *M
fHRF
D__inference_conv2d_2_layer_call_and_return_conditional_losses_1437952
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:�����������@2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:����������� ::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:����������� 
 
_user_specified_nameinputs
�
~
)__inference_conv2d_4_layer_call_fn_144375

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *M
fHRF
D__inference_conv2d_4_layer_call_and_return_conditional_losses_1440962
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+���������������������������::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�

�
B__inference_conv2d_layer_call_and_return_conditional_losses_143741

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������*
paddingSAME*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������2	
BiasAddb
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:�����������2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:�����������2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:�����������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
ݦ
�0
"__inference__traced_restore_144786
file_prefix
assignvariableop_adam_iter"
assignvariableop_1_adam_beta_1"
assignvariableop_2_adam_beta_2!
assignvariableop_3_adam_decay)
%assignvariableop_4_adam_learning_rateG
Cassignvariableop_5_bigger_auto_encoder_bigger_encoder_conv2d_kernelE
Aassignvariableop_6_bigger_auto_encoder_bigger_encoder_conv2d_biasI
Eassignvariableop_7_bigger_auto_encoder_bigger_encoder_conv2d_1_kernelG
Cassignvariableop_8_bigger_auto_encoder_bigger_encoder_conv2d_1_biasI
Eassignvariableop_9_bigger_auto_encoder_bigger_encoder_conv2d_2_kernelH
Dassignvariableop_10_bigger_auto_encoder_bigger_encoder_conv2d_2_biasJ
Fassignvariableop_11_bigger_auto_encoder_bigger_encoder_conv2d_3_kernelH
Dassignvariableop_12_bigger_auto_encoder_bigger_encoder_conv2d_3_biasR
Nassignvariableop_13_bigger_auto_encoder_bigger_decoder_conv2d_transpose_kernelP
Lassignvariableop_14_bigger_auto_encoder_bigger_decoder_conv2d_transpose_biasT
Passignvariableop_15_bigger_auto_encoder_bigger_decoder_conv2d_transpose_1_kernelR
Nassignvariableop_16_bigger_auto_encoder_bigger_decoder_conv2d_transpose_1_biasT
Passignvariableop_17_bigger_auto_encoder_bigger_decoder_conv2d_transpose_2_kernelR
Nassignvariableop_18_bigger_auto_encoder_bigger_decoder_conv2d_transpose_2_biasT
Passignvariableop_19_bigger_auto_encoder_bigger_decoder_conv2d_transpose_3_kernelR
Nassignvariableop_20_bigger_auto_encoder_bigger_decoder_conv2d_transpose_3_biasJ
Fassignvariableop_21_bigger_auto_encoder_bigger_decoder_conv2d_4_kernelH
Dassignvariableop_22_bigger_auto_encoder_bigger_decoder_conv2d_4_bias
assignvariableop_23_total
assignvariableop_24_count
assignvariableop_25_total_1
assignvariableop_26_count_1O
Kassignvariableop_27_adam_bigger_auto_encoder_bigger_encoder_conv2d_kernel_mM
Iassignvariableop_28_adam_bigger_auto_encoder_bigger_encoder_conv2d_bias_mQ
Massignvariableop_29_adam_bigger_auto_encoder_bigger_encoder_conv2d_1_kernel_mO
Kassignvariableop_30_adam_bigger_auto_encoder_bigger_encoder_conv2d_1_bias_mQ
Massignvariableop_31_adam_bigger_auto_encoder_bigger_encoder_conv2d_2_kernel_mO
Kassignvariableop_32_adam_bigger_auto_encoder_bigger_encoder_conv2d_2_bias_mQ
Massignvariableop_33_adam_bigger_auto_encoder_bigger_encoder_conv2d_3_kernel_mO
Kassignvariableop_34_adam_bigger_auto_encoder_bigger_encoder_conv2d_3_bias_mY
Uassignvariableop_35_adam_bigger_auto_encoder_bigger_decoder_conv2d_transpose_kernel_mW
Sassignvariableop_36_adam_bigger_auto_encoder_bigger_decoder_conv2d_transpose_bias_m[
Wassignvariableop_37_adam_bigger_auto_encoder_bigger_decoder_conv2d_transpose_1_kernel_mY
Uassignvariableop_38_adam_bigger_auto_encoder_bigger_decoder_conv2d_transpose_1_bias_m[
Wassignvariableop_39_adam_bigger_auto_encoder_bigger_decoder_conv2d_transpose_2_kernel_mY
Uassignvariableop_40_adam_bigger_auto_encoder_bigger_decoder_conv2d_transpose_2_bias_m[
Wassignvariableop_41_adam_bigger_auto_encoder_bigger_decoder_conv2d_transpose_3_kernel_mY
Uassignvariableop_42_adam_bigger_auto_encoder_bigger_decoder_conv2d_transpose_3_bias_mQ
Massignvariableop_43_adam_bigger_auto_encoder_bigger_decoder_conv2d_4_kernel_mO
Kassignvariableop_44_adam_bigger_auto_encoder_bigger_decoder_conv2d_4_bias_mO
Kassignvariableop_45_adam_bigger_auto_encoder_bigger_encoder_conv2d_kernel_vM
Iassignvariableop_46_adam_bigger_auto_encoder_bigger_encoder_conv2d_bias_vQ
Massignvariableop_47_adam_bigger_auto_encoder_bigger_encoder_conv2d_1_kernel_vO
Kassignvariableop_48_adam_bigger_auto_encoder_bigger_encoder_conv2d_1_bias_vQ
Massignvariableop_49_adam_bigger_auto_encoder_bigger_encoder_conv2d_2_kernel_vO
Kassignvariableop_50_adam_bigger_auto_encoder_bigger_encoder_conv2d_2_bias_vQ
Massignvariableop_51_adam_bigger_auto_encoder_bigger_encoder_conv2d_3_kernel_vO
Kassignvariableop_52_adam_bigger_auto_encoder_bigger_encoder_conv2d_3_bias_vY
Uassignvariableop_53_adam_bigger_auto_encoder_bigger_decoder_conv2d_transpose_kernel_vW
Sassignvariableop_54_adam_bigger_auto_encoder_bigger_decoder_conv2d_transpose_bias_v[
Wassignvariableop_55_adam_bigger_auto_encoder_bigger_decoder_conv2d_transpose_1_kernel_vY
Uassignvariableop_56_adam_bigger_auto_encoder_bigger_decoder_conv2d_transpose_1_bias_v[
Wassignvariableop_57_adam_bigger_auto_encoder_bigger_decoder_conv2d_transpose_2_kernel_vY
Uassignvariableop_58_adam_bigger_auto_encoder_bigger_decoder_conv2d_transpose_2_bias_v[
Wassignvariableop_59_adam_bigger_auto_encoder_bigger_decoder_conv2d_transpose_3_kernel_vY
Uassignvariableop_60_adam_bigger_auto_encoder_bigger_decoder_conv2d_transpose_3_bias_vQ
Massignvariableop_61_adam_bigger_auto_encoder_bigger_decoder_conv2d_4_kernel_vO
Kassignvariableop_62_adam_bigger_auto_encoder_bigger_decoder_conv2d_4_bias_v
identity_64��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_25�AssignVariableOp_26�AssignVariableOp_27�AssignVariableOp_28�AssignVariableOp_29�AssignVariableOp_3�AssignVariableOp_30�AssignVariableOp_31�AssignVariableOp_32�AssignVariableOp_33�AssignVariableOp_34�AssignVariableOp_35�AssignVariableOp_36�AssignVariableOp_37�AssignVariableOp_38�AssignVariableOp_39�AssignVariableOp_4�AssignVariableOp_40�AssignVariableOp_41�AssignVariableOp_42�AssignVariableOp_43�AssignVariableOp_44�AssignVariableOp_45�AssignVariableOp_46�AssignVariableOp_47�AssignVariableOp_48�AssignVariableOp_49�AssignVariableOp_5�AssignVariableOp_50�AssignVariableOp_51�AssignVariableOp_52�AssignVariableOp_53�AssignVariableOp_54�AssignVariableOp_55�AssignVariableOp_56�AssignVariableOp_57�AssignVariableOp_58�AssignVariableOp_59�AssignVariableOp_6�AssignVariableOp_60�AssignVariableOp_61�AssignVariableOp_62�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�!
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:@*
dtype0*�!
value� B� @B)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/11/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/12/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/13/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/14/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/15/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/16/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/17/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:@*
dtype0*�
value�B�@B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices�
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*�
_output_shapes�
�::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*N
dtypesD
B2@	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0	*
_output_shapes
:2

Identity�
AssignVariableOpAssignVariableOpassignvariableop_adam_iterIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1�
AssignVariableOp_1AssignVariableOpassignvariableop_1_adam_beta_1Identity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2�
AssignVariableOp_2AssignVariableOpassignvariableop_2_adam_beta_2Identity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3�
AssignVariableOp_3AssignVariableOpassignvariableop_3_adam_decayIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4�
AssignVariableOp_4AssignVariableOp%assignvariableop_4_adam_learning_rateIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5�
AssignVariableOp_5AssignVariableOpCassignvariableop_5_bigger_auto_encoder_bigger_encoder_conv2d_kernelIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6�
AssignVariableOp_6AssignVariableOpAassignvariableop_6_bigger_auto_encoder_bigger_encoder_conv2d_biasIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7�
AssignVariableOp_7AssignVariableOpEassignvariableop_7_bigger_auto_encoder_bigger_encoder_conv2d_1_kernelIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8�
AssignVariableOp_8AssignVariableOpCassignvariableop_8_bigger_auto_encoder_bigger_encoder_conv2d_1_biasIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9�
AssignVariableOp_9AssignVariableOpEassignvariableop_9_bigger_auto_encoder_bigger_encoder_conv2d_2_kernelIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10�
AssignVariableOp_10AssignVariableOpDassignvariableop_10_bigger_auto_encoder_bigger_encoder_conv2d_2_biasIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11�
AssignVariableOp_11AssignVariableOpFassignvariableop_11_bigger_auto_encoder_bigger_encoder_conv2d_3_kernelIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12�
AssignVariableOp_12AssignVariableOpDassignvariableop_12_bigger_auto_encoder_bigger_encoder_conv2d_3_biasIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13�
AssignVariableOp_13AssignVariableOpNassignvariableop_13_bigger_auto_encoder_bigger_decoder_conv2d_transpose_kernelIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14�
AssignVariableOp_14AssignVariableOpLassignvariableop_14_bigger_auto_encoder_bigger_decoder_conv2d_transpose_biasIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15�
AssignVariableOp_15AssignVariableOpPassignvariableop_15_bigger_auto_encoder_bigger_decoder_conv2d_transpose_1_kernelIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16�
AssignVariableOp_16AssignVariableOpNassignvariableop_16_bigger_auto_encoder_bigger_decoder_conv2d_transpose_1_biasIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17�
AssignVariableOp_17AssignVariableOpPassignvariableop_17_bigger_auto_encoder_bigger_decoder_conv2d_transpose_2_kernelIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18�
AssignVariableOp_18AssignVariableOpNassignvariableop_18_bigger_auto_encoder_bigger_decoder_conv2d_transpose_2_biasIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19�
AssignVariableOp_19AssignVariableOpPassignvariableop_19_bigger_auto_encoder_bigger_decoder_conv2d_transpose_3_kernelIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20�
AssignVariableOp_20AssignVariableOpNassignvariableop_20_bigger_auto_encoder_bigger_decoder_conv2d_transpose_3_biasIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21�
AssignVariableOp_21AssignVariableOpFassignvariableop_21_bigger_auto_encoder_bigger_decoder_conv2d_4_kernelIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22�
AssignVariableOp_22AssignVariableOpDassignvariableop_22_bigger_auto_encoder_bigger_decoder_conv2d_4_biasIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23�
AssignVariableOp_23AssignVariableOpassignvariableop_23_totalIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24�
AssignVariableOp_24AssignVariableOpassignvariableop_24_countIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25�
AssignVariableOp_25AssignVariableOpassignvariableop_25_total_1Identity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26�
AssignVariableOp_26AssignVariableOpassignvariableop_26_count_1Identity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27�
AssignVariableOp_27AssignVariableOpKassignvariableop_27_adam_bigger_auto_encoder_bigger_encoder_conv2d_kernel_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28�
AssignVariableOp_28AssignVariableOpIassignvariableop_28_adam_bigger_auto_encoder_bigger_encoder_conv2d_bias_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29�
AssignVariableOp_29AssignVariableOpMassignvariableop_29_adam_bigger_auto_encoder_bigger_encoder_conv2d_1_kernel_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30�
AssignVariableOp_30AssignVariableOpKassignvariableop_30_adam_bigger_auto_encoder_bigger_encoder_conv2d_1_bias_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31�
AssignVariableOp_31AssignVariableOpMassignvariableop_31_adam_bigger_auto_encoder_bigger_encoder_conv2d_2_kernel_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32�
AssignVariableOp_32AssignVariableOpKassignvariableop_32_adam_bigger_auto_encoder_bigger_encoder_conv2d_2_bias_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33�
AssignVariableOp_33AssignVariableOpMassignvariableop_33_adam_bigger_auto_encoder_bigger_encoder_conv2d_3_kernel_mIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34�
AssignVariableOp_34AssignVariableOpKassignvariableop_34_adam_bigger_auto_encoder_bigger_encoder_conv2d_3_bias_mIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35�
AssignVariableOp_35AssignVariableOpUassignvariableop_35_adam_bigger_auto_encoder_bigger_decoder_conv2d_transpose_kernel_mIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36�
AssignVariableOp_36AssignVariableOpSassignvariableop_36_adam_bigger_auto_encoder_bigger_decoder_conv2d_transpose_bias_mIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37�
AssignVariableOp_37AssignVariableOpWassignvariableop_37_adam_bigger_auto_encoder_bigger_decoder_conv2d_transpose_1_kernel_mIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38�
AssignVariableOp_38AssignVariableOpUassignvariableop_38_adam_bigger_auto_encoder_bigger_decoder_conv2d_transpose_1_bias_mIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39�
AssignVariableOp_39AssignVariableOpWassignvariableop_39_adam_bigger_auto_encoder_bigger_decoder_conv2d_transpose_2_kernel_mIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40�
AssignVariableOp_40AssignVariableOpUassignvariableop_40_adam_bigger_auto_encoder_bigger_decoder_conv2d_transpose_2_bias_mIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41�
AssignVariableOp_41AssignVariableOpWassignvariableop_41_adam_bigger_auto_encoder_bigger_decoder_conv2d_transpose_3_kernel_mIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42�
AssignVariableOp_42AssignVariableOpUassignvariableop_42_adam_bigger_auto_encoder_bigger_decoder_conv2d_transpose_3_bias_mIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43�
AssignVariableOp_43AssignVariableOpMassignvariableop_43_adam_bigger_auto_encoder_bigger_decoder_conv2d_4_kernel_mIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44�
AssignVariableOp_44AssignVariableOpKassignvariableop_44_adam_bigger_auto_encoder_bigger_decoder_conv2d_4_bias_mIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45�
AssignVariableOp_45AssignVariableOpKassignvariableop_45_adam_bigger_auto_encoder_bigger_encoder_conv2d_kernel_vIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46�
AssignVariableOp_46AssignVariableOpIassignvariableop_46_adam_bigger_auto_encoder_bigger_encoder_conv2d_bias_vIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47�
AssignVariableOp_47AssignVariableOpMassignvariableop_47_adam_bigger_auto_encoder_bigger_encoder_conv2d_1_kernel_vIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48�
AssignVariableOp_48AssignVariableOpKassignvariableop_48_adam_bigger_auto_encoder_bigger_encoder_conv2d_1_bias_vIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49�
AssignVariableOp_49AssignVariableOpMassignvariableop_49_adam_bigger_auto_encoder_bigger_encoder_conv2d_2_kernel_vIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50�
AssignVariableOp_50AssignVariableOpKassignvariableop_50_adam_bigger_auto_encoder_bigger_encoder_conv2d_2_bias_vIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_50n
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:2
Identity_51�
AssignVariableOp_51AssignVariableOpMassignvariableop_51_adam_bigger_auto_encoder_bigger_encoder_conv2d_3_kernel_vIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_51n
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:2
Identity_52�
AssignVariableOp_52AssignVariableOpKassignvariableop_52_adam_bigger_auto_encoder_bigger_encoder_conv2d_3_bias_vIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_52n
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:2
Identity_53�
AssignVariableOp_53AssignVariableOpUassignvariableop_53_adam_bigger_auto_encoder_bigger_decoder_conv2d_transpose_kernel_vIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_53n
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:2
Identity_54�
AssignVariableOp_54AssignVariableOpSassignvariableop_54_adam_bigger_auto_encoder_bigger_decoder_conv2d_transpose_bias_vIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_54n
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:2
Identity_55�
AssignVariableOp_55AssignVariableOpWassignvariableop_55_adam_bigger_auto_encoder_bigger_decoder_conv2d_transpose_1_kernel_vIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_55n
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:2
Identity_56�
AssignVariableOp_56AssignVariableOpUassignvariableop_56_adam_bigger_auto_encoder_bigger_decoder_conv2d_transpose_1_bias_vIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_56n
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:2
Identity_57�
AssignVariableOp_57AssignVariableOpWassignvariableop_57_adam_bigger_auto_encoder_bigger_decoder_conv2d_transpose_2_kernel_vIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_57n
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:2
Identity_58�
AssignVariableOp_58AssignVariableOpUassignvariableop_58_adam_bigger_auto_encoder_bigger_decoder_conv2d_transpose_2_bias_vIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_58n
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:2
Identity_59�
AssignVariableOp_59AssignVariableOpWassignvariableop_59_adam_bigger_auto_encoder_bigger_decoder_conv2d_transpose_3_kernel_vIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_59n
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:2
Identity_60�
AssignVariableOp_60AssignVariableOpUassignvariableop_60_adam_bigger_auto_encoder_bigger_decoder_conv2d_transpose_3_bias_vIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_60n
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:2
Identity_61�
AssignVariableOp_61AssignVariableOpMassignvariableop_61_adam_bigger_auto_encoder_bigger_decoder_conv2d_4_kernel_vIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_61n
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:2
Identity_62�
AssignVariableOp_62AssignVariableOpKassignvariableop_62_adam_bigger_auto_encoder_bigger_decoder_conv2d_4_bias_vIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_629
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp�
Identity_63Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_63�
Identity_64IdentityIdentity_63:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_64"#
identity_64Identity_64:output:0*�
_input_shapes�
�: :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�
~
)__inference_conv2d_3_layer_call_fn_144355

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *M
fHRF
D__inference_conv2d_3_layer_call_and_return_conditional_losses_1438222
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:�����������@2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:�����������@::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:�����������@
 
_user_specified_nameinputs
�$
�
N__inference_conv2d_transpose_2_layer_call_and_return_conditional_losses_143986

inputs,
(conv2d_transpose_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�conv2d_transpose/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2�
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1x
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2�
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2P
mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
mul/y\
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mulT
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
mul_1/yb
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: 2
mul_1T
stack/3Const*
_output_shapes
: *
dtype0*
value	B : 2	
stack/3�
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:2
stackx
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2�
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3�
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
: @*
dtype02!
conv2d_transpose/ReadVariableOp�
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+��������������������������� *
paddingSAME*
strides
2
conv2d_transpose�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+��������������������������� 2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+��������������������������� 2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*
T0*A
_output_shapes/
-:+��������������������������� 2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+���������������������������@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+���������������������������@
 
_user_specified_nameinputs
�$
�
N__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_144031

inputs,
(conv2d_transpose_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�conv2d_transpose/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2�
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1x
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2�
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2P
mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
mul/y\
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mulT
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
mul_1/yb
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: 2
mul_1T
stack/3Const*
_output_shapes
: *
dtype0*
value	B :2	
stack/3�
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:2
stackx
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2�
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3�
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_transpose/ReadVariableOp�
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+���������������������������*
paddingSAME*
strides
2
conv2d_transpose�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+���������������������������2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+���������������������������2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+��������������������������� ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+��������������������������� 
 
_user_specified_nameinputs
�
�
3__inference_conv2d_transpose_2_layer_call_fn_143996

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+��������������������������� *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *W
fRRP
N__inference_conv2d_transpose_2_layer_call_and_return_conditional_losses_1439862
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+��������������������������� 2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+���������������������������@::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+���������������������������@
 
_user_specified_nameinputs
�
�
1__inference_conv2d_transpose_layer_call_fn_143906

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *U
fPRN
L__inference_conv2d_transpose_layer_call_and_return_conditional_losses_1438962
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������@2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+���������������������������@::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+���������������������������@
 
_user_specified_nameinputs
ɲ
�
!__inference__wrapped_model_143726
input_1L
Hbigger_auto_encoder_bigger_encoder_conv2d_conv2d_readvariableop_resourceM
Ibigger_auto_encoder_bigger_encoder_conv2d_biasadd_readvariableop_resourceN
Jbigger_auto_encoder_bigger_encoder_conv2d_1_conv2d_readvariableop_resourceO
Kbigger_auto_encoder_bigger_encoder_conv2d_1_biasadd_readvariableop_resourceN
Jbigger_auto_encoder_bigger_encoder_conv2d_2_conv2d_readvariableop_resourceO
Kbigger_auto_encoder_bigger_encoder_conv2d_2_biasadd_readvariableop_resourceN
Jbigger_auto_encoder_bigger_encoder_conv2d_3_conv2d_readvariableop_resourceO
Kbigger_auto_encoder_bigger_encoder_conv2d_3_biasadd_readvariableop_resource`
\bigger_auto_encoder_bigger_decoder_conv2d_transpose_conv2d_transpose_readvariableop_resourceW
Sbigger_auto_encoder_bigger_decoder_conv2d_transpose_biasadd_readvariableop_resourceb
^bigger_auto_encoder_bigger_decoder_conv2d_transpose_1_conv2d_transpose_readvariableop_resourceY
Ubigger_auto_encoder_bigger_decoder_conv2d_transpose_1_biasadd_readvariableop_resourceb
^bigger_auto_encoder_bigger_decoder_conv2d_transpose_2_conv2d_transpose_readvariableop_resourceY
Ubigger_auto_encoder_bigger_decoder_conv2d_transpose_2_biasadd_readvariableop_resourceb
^bigger_auto_encoder_bigger_decoder_conv2d_transpose_3_conv2d_transpose_readvariableop_resourceY
Ubigger_auto_encoder_bigger_decoder_conv2d_transpose_3_biasadd_readvariableop_resourceN
Jbigger_auto_encoder_bigger_decoder_conv2d_4_conv2d_readvariableop_resourceO
Kbigger_auto_encoder_bigger_decoder_conv2d_4_biasadd_readvariableop_resource
identity��Bbigger_auto_encoder/bigger_decoder/conv2d_4/BiasAdd/ReadVariableOp�Abigger_auto_encoder/bigger_decoder/conv2d_4/Conv2D/ReadVariableOp�Jbigger_auto_encoder/bigger_decoder/conv2d_transpose/BiasAdd/ReadVariableOp�Sbigger_auto_encoder/bigger_decoder/conv2d_transpose/conv2d_transpose/ReadVariableOp�Lbigger_auto_encoder/bigger_decoder/conv2d_transpose_1/BiasAdd/ReadVariableOp�Ubigger_auto_encoder/bigger_decoder/conv2d_transpose_1/conv2d_transpose/ReadVariableOp�Lbigger_auto_encoder/bigger_decoder/conv2d_transpose_2/BiasAdd/ReadVariableOp�Ubigger_auto_encoder/bigger_decoder/conv2d_transpose_2/conv2d_transpose/ReadVariableOp�Lbigger_auto_encoder/bigger_decoder/conv2d_transpose_3/BiasAdd/ReadVariableOp�Ubigger_auto_encoder/bigger_decoder/conv2d_transpose_3/conv2d_transpose/ReadVariableOp�@bigger_auto_encoder/bigger_encoder/conv2d/BiasAdd/ReadVariableOp�?bigger_auto_encoder/bigger_encoder/conv2d/Conv2D/ReadVariableOp�Bbigger_auto_encoder/bigger_encoder/conv2d_1/BiasAdd/ReadVariableOp�Abigger_auto_encoder/bigger_encoder/conv2d_1/Conv2D/ReadVariableOp�Bbigger_auto_encoder/bigger_encoder/conv2d_2/BiasAdd/ReadVariableOp�Abigger_auto_encoder/bigger_encoder/conv2d_2/Conv2D/ReadVariableOp�Bbigger_auto_encoder/bigger_encoder/conv2d_3/BiasAdd/ReadVariableOp�Abigger_auto_encoder/bigger_encoder/conv2d_3/Conv2D/ReadVariableOp�
?bigger_auto_encoder/bigger_encoder/conv2d/Conv2D/ReadVariableOpReadVariableOpHbigger_auto_encoder_bigger_encoder_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02A
?bigger_auto_encoder/bigger_encoder/conv2d/Conv2D/ReadVariableOp�
0bigger_auto_encoder/bigger_encoder/conv2d/Conv2DConv2Dinput_1Gbigger_auto_encoder/bigger_encoder/conv2d/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������*
paddingSAME*
strides
22
0bigger_auto_encoder/bigger_encoder/conv2d/Conv2D�
@bigger_auto_encoder/bigger_encoder/conv2d/BiasAdd/ReadVariableOpReadVariableOpIbigger_auto_encoder_bigger_encoder_conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02B
@bigger_auto_encoder/bigger_encoder/conv2d/BiasAdd/ReadVariableOp�
1bigger_auto_encoder/bigger_encoder/conv2d/BiasAddBiasAdd9bigger_auto_encoder/bigger_encoder/conv2d/Conv2D:output:0Hbigger_auto_encoder/bigger_encoder/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������23
1bigger_auto_encoder/bigger_encoder/conv2d/BiasAdd�
.bigger_auto_encoder/bigger_encoder/conv2d/ReluRelu:bigger_auto_encoder/bigger_encoder/conv2d/BiasAdd:output:0*
T0*1
_output_shapes
:�����������20
.bigger_auto_encoder/bigger_encoder/conv2d/Relu�
Abigger_auto_encoder/bigger_encoder/conv2d_1/Conv2D/ReadVariableOpReadVariableOpJbigger_auto_encoder_bigger_encoder_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02C
Abigger_auto_encoder/bigger_encoder/conv2d_1/Conv2D/ReadVariableOp�
2bigger_auto_encoder/bigger_encoder/conv2d_1/Conv2DConv2D<bigger_auto_encoder/bigger_encoder/conv2d/Relu:activations:0Ibigger_auto_encoder/bigger_encoder/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:����������� *
paddingSAME*
strides
24
2bigger_auto_encoder/bigger_encoder/conv2d_1/Conv2D�
Bbigger_auto_encoder/bigger_encoder/conv2d_1/BiasAdd/ReadVariableOpReadVariableOpKbigger_auto_encoder_bigger_encoder_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02D
Bbigger_auto_encoder/bigger_encoder/conv2d_1/BiasAdd/ReadVariableOp�
3bigger_auto_encoder/bigger_encoder/conv2d_1/BiasAddBiasAdd;bigger_auto_encoder/bigger_encoder/conv2d_1/Conv2D:output:0Jbigger_auto_encoder/bigger_encoder/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:����������� 25
3bigger_auto_encoder/bigger_encoder/conv2d_1/BiasAdd�
0bigger_auto_encoder/bigger_encoder/conv2d_1/ReluRelu<bigger_auto_encoder/bigger_encoder/conv2d_1/BiasAdd:output:0*
T0*1
_output_shapes
:����������� 22
0bigger_auto_encoder/bigger_encoder/conv2d_1/Relu�
Abigger_auto_encoder/bigger_encoder/conv2d_2/Conv2D/ReadVariableOpReadVariableOpJbigger_auto_encoder_bigger_encoder_conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02C
Abigger_auto_encoder/bigger_encoder/conv2d_2/Conv2D/ReadVariableOp�
2bigger_auto_encoder/bigger_encoder/conv2d_2/Conv2DConv2D>bigger_auto_encoder/bigger_encoder/conv2d_1/Relu:activations:0Ibigger_auto_encoder/bigger_encoder/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������@*
paddingSAME*
strides
24
2bigger_auto_encoder/bigger_encoder/conv2d_2/Conv2D�
Bbigger_auto_encoder/bigger_encoder/conv2d_2/BiasAdd/ReadVariableOpReadVariableOpKbigger_auto_encoder_bigger_encoder_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02D
Bbigger_auto_encoder/bigger_encoder/conv2d_2/BiasAdd/ReadVariableOp�
3bigger_auto_encoder/bigger_encoder/conv2d_2/BiasAddBiasAdd;bigger_auto_encoder/bigger_encoder/conv2d_2/Conv2D:output:0Jbigger_auto_encoder/bigger_encoder/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������@25
3bigger_auto_encoder/bigger_encoder/conv2d_2/BiasAdd�
0bigger_auto_encoder/bigger_encoder/conv2d_2/ReluRelu<bigger_auto_encoder/bigger_encoder/conv2d_2/BiasAdd:output:0*
T0*1
_output_shapes
:�����������@22
0bigger_auto_encoder/bigger_encoder/conv2d_2/Relu�
Abigger_auto_encoder/bigger_encoder/conv2d_3/Conv2D/ReadVariableOpReadVariableOpJbigger_auto_encoder_bigger_encoder_conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02C
Abigger_auto_encoder/bigger_encoder/conv2d_3/Conv2D/ReadVariableOp�
2bigger_auto_encoder/bigger_encoder/conv2d_3/Conv2DConv2D>bigger_auto_encoder/bigger_encoder/conv2d_2/Relu:activations:0Ibigger_auto_encoder/bigger_encoder/conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������@*
paddingSAME*
strides
24
2bigger_auto_encoder/bigger_encoder/conv2d_3/Conv2D�
Bbigger_auto_encoder/bigger_encoder/conv2d_3/BiasAdd/ReadVariableOpReadVariableOpKbigger_auto_encoder_bigger_encoder_conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02D
Bbigger_auto_encoder/bigger_encoder/conv2d_3/BiasAdd/ReadVariableOp�
3bigger_auto_encoder/bigger_encoder/conv2d_3/BiasAddBiasAdd;bigger_auto_encoder/bigger_encoder/conv2d_3/Conv2D:output:0Jbigger_auto_encoder/bigger_encoder/conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������@25
3bigger_auto_encoder/bigger_encoder/conv2d_3/BiasAdd�
0bigger_auto_encoder/bigger_encoder/conv2d_3/ReluRelu<bigger_auto_encoder/bigger_encoder/conv2d_3/BiasAdd:output:0*
T0*1
_output_shapes
:�����������@22
0bigger_auto_encoder/bigger_encoder/conv2d_3/Relu�
9bigger_auto_encoder/bigger_decoder/conv2d_transpose/ShapeShape>bigger_auto_encoder/bigger_encoder/conv2d_3/Relu:activations:0*
T0*
_output_shapes
:2;
9bigger_auto_encoder/bigger_decoder/conv2d_transpose/Shape�
Gbigger_auto_encoder/bigger_decoder/conv2d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2I
Gbigger_auto_encoder/bigger_decoder/conv2d_transpose/strided_slice/stack�
Ibigger_auto_encoder/bigger_decoder/conv2d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2K
Ibigger_auto_encoder/bigger_decoder/conv2d_transpose/strided_slice/stack_1�
Ibigger_auto_encoder/bigger_decoder/conv2d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2K
Ibigger_auto_encoder/bigger_decoder/conv2d_transpose/strided_slice/stack_2�
Abigger_auto_encoder/bigger_decoder/conv2d_transpose/strided_sliceStridedSliceBbigger_auto_encoder/bigger_decoder/conv2d_transpose/Shape:output:0Pbigger_auto_encoder/bigger_decoder/conv2d_transpose/strided_slice/stack:output:0Rbigger_auto_encoder/bigger_decoder/conv2d_transpose/strided_slice/stack_1:output:0Rbigger_auto_encoder/bigger_decoder/conv2d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2C
Abigger_auto_encoder/bigger_decoder/conv2d_transpose/strided_slice�
;bigger_auto_encoder/bigger_decoder/conv2d_transpose/stack/1Const*
_output_shapes
: *
dtype0*
value
B :�2=
;bigger_auto_encoder/bigger_decoder/conv2d_transpose/stack/1�
;bigger_auto_encoder/bigger_decoder/conv2d_transpose/stack/2Const*
_output_shapes
: *
dtype0*
value
B :�2=
;bigger_auto_encoder/bigger_decoder/conv2d_transpose/stack/2�
;bigger_auto_encoder/bigger_decoder/conv2d_transpose/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@2=
;bigger_auto_encoder/bigger_decoder/conv2d_transpose/stack/3�
9bigger_auto_encoder/bigger_decoder/conv2d_transpose/stackPackJbigger_auto_encoder/bigger_decoder/conv2d_transpose/strided_slice:output:0Dbigger_auto_encoder/bigger_decoder/conv2d_transpose/stack/1:output:0Dbigger_auto_encoder/bigger_decoder/conv2d_transpose/stack/2:output:0Dbigger_auto_encoder/bigger_decoder/conv2d_transpose/stack/3:output:0*
N*
T0*
_output_shapes
:2;
9bigger_auto_encoder/bigger_decoder/conv2d_transpose/stack�
Ibigger_auto_encoder/bigger_decoder/conv2d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2K
Ibigger_auto_encoder/bigger_decoder/conv2d_transpose/strided_slice_1/stack�
Kbigger_auto_encoder/bigger_decoder/conv2d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2M
Kbigger_auto_encoder/bigger_decoder/conv2d_transpose/strided_slice_1/stack_1�
Kbigger_auto_encoder/bigger_decoder/conv2d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2M
Kbigger_auto_encoder/bigger_decoder/conv2d_transpose/strided_slice_1/stack_2�
Cbigger_auto_encoder/bigger_decoder/conv2d_transpose/strided_slice_1StridedSliceBbigger_auto_encoder/bigger_decoder/conv2d_transpose/stack:output:0Rbigger_auto_encoder/bigger_decoder/conv2d_transpose/strided_slice_1/stack:output:0Tbigger_auto_encoder/bigger_decoder/conv2d_transpose/strided_slice_1/stack_1:output:0Tbigger_auto_encoder/bigger_decoder/conv2d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2E
Cbigger_auto_encoder/bigger_decoder/conv2d_transpose/strided_slice_1�
Sbigger_auto_encoder/bigger_decoder/conv2d_transpose/conv2d_transpose/ReadVariableOpReadVariableOp\bigger_auto_encoder_bigger_decoder_conv2d_transpose_conv2d_transpose_readvariableop_resource*&
_output_shapes
:@@*
dtype02U
Sbigger_auto_encoder/bigger_decoder/conv2d_transpose/conv2d_transpose/ReadVariableOp�
Dbigger_auto_encoder/bigger_decoder/conv2d_transpose/conv2d_transposeConv2DBackpropInputBbigger_auto_encoder/bigger_decoder/conv2d_transpose/stack:output:0[bigger_auto_encoder/bigger_decoder/conv2d_transpose/conv2d_transpose/ReadVariableOp:value:0>bigger_auto_encoder/bigger_encoder/conv2d_3/Relu:activations:0*
T0*1
_output_shapes
:�����������@*
paddingSAME*
strides
2F
Dbigger_auto_encoder/bigger_decoder/conv2d_transpose/conv2d_transpose�
Jbigger_auto_encoder/bigger_decoder/conv2d_transpose/BiasAdd/ReadVariableOpReadVariableOpSbigger_auto_encoder_bigger_decoder_conv2d_transpose_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02L
Jbigger_auto_encoder/bigger_decoder/conv2d_transpose/BiasAdd/ReadVariableOp�
;bigger_auto_encoder/bigger_decoder/conv2d_transpose/BiasAddBiasAddMbigger_auto_encoder/bigger_decoder/conv2d_transpose/conv2d_transpose:output:0Rbigger_auto_encoder/bigger_decoder/conv2d_transpose/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������@2=
;bigger_auto_encoder/bigger_decoder/conv2d_transpose/BiasAdd�
8bigger_auto_encoder/bigger_decoder/conv2d_transpose/ReluReluDbigger_auto_encoder/bigger_decoder/conv2d_transpose/BiasAdd:output:0*
T0*1
_output_shapes
:�����������@2:
8bigger_auto_encoder/bigger_decoder/conv2d_transpose/Relu�
;bigger_auto_encoder/bigger_decoder/conv2d_transpose_1/ShapeShapeFbigger_auto_encoder/bigger_decoder/conv2d_transpose/Relu:activations:0*
T0*
_output_shapes
:2=
;bigger_auto_encoder/bigger_decoder/conv2d_transpose_1/Shape�
Ibigger_auto_encoder/bigger_decoder/conv2d_transpose_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2K
Ibigger_auto_encoder/bigger_decoder/conv2d_transpose_1/strided_slice/stack�
Kbigger_auto_encoder/bigger_decoder/conv2d_transpose_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2M
Kbigger_auto_encoder/bigger_decoder/conv2d_transpose_1/strided_slice/stack_1�
Kbigger_auto_encoder/bigger_decoder/conv2d_transpose_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2M
Kbigger_auto_encoder/bigger_decoder/conv2d_transpose_1/strided_slice/stack_2�
Cbigger_auto_encoder/bigger_decoder/conv2d_transpose_1/strided_sliceStridedSliceDbigger_auto_encoder/bigger_decoder/conv2d_transpose_1/Shape:output:0Rbigger_auto_encoder/bigger_decoder/conv2d_transpose_1/strided_slice/stack:output:0Tbigger_auto_encoder/bigger_decoder/conv2d_transpose_1/strided_slice/stack_1:output:0Tbigger_auto_encoder/bigger_decoder/conv2d_transpose_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2E
Cbigger_auto_encoder/bigger_decoder/conv2d_transpose_1/strided_slice�
=bigger_auto_encoder/bigger_decoder/conv2d_transpose_1/stack/1Const*
_output_shapes
: *
dtype0*
value
B :�2?
=bigger_auto_encoder/bigger_decoder/conv2d_transpose_1/stack/1�
=bigger_auto_encoder/bigger_decoder/conv2d_transpose_1/stack/2Const*
_output_shapes
: *
dtype0*
value
B :�2?
=bigger_auto_encoder/bigger_decoder/conv2d_transpose_1/stack/2�
=bigger_auto_encoder/bigger_decoder/conv2d_transpose_1/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@2?
=bigger_auto_encoder/bigger_decoder/conv2d_transpose_1/stack/3�
;bigger_auto_encoder/bigger_decoder/conv2d_transpose_1/stackPackLbigger_auto_encoder/bigger_decoder/conv2d_transpose_1/strided_slice:output:0Fbigger_auto_encoder/bigger_decoder/conv2d_transpose_1/stack/1:output:0Fbigger_auto_encoder/bigger_decoder/conv2d_transpose_1/stack/2:output:0Fbigger_auto_encoder/bigger_decoder/conv2d_transpose_1/stack/3:output:0*
N*
T0*
_output_shapes
:2=
;bigger_auto_encoder/bigger_decoder/conv2d_transpose_1/stack�
Kbigger_auto_encoder/bigger_decoder/conv2d_transpose_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2M
Kbigger_auto_encoder/bigger_decoder/conv2d_transpose_1/strided_slice_1/stack�
Mbigger_auto_encoder/bigger_decoder/conv2d_transpose_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2O
Mbigger_auto_encoder/bigger_decoder/conv2d_transpose_1/strided_slice_1/stack_1�
Mbigger_auto_encoder/bigger_decoder/conv2d_transpose_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2O
Mbigger_auto_encoder/bigger_decoder/conv2d_transpose_1/strided_slice_1/stack_2�
Ebigger_auto_encoder/bigger_decoder/conv2d_transpose_1/strided_slice_1StridedSliceDbigger_auto_encoder/bigger_decoder/conv2d_transpose_1/stack:output:0Tbigger_auto_encoder/bigger_decoder/conv2d_transpose_1/strided_slice_1/stack:output:0Vbigger_auto_encoder/bigger_decoder/conv2d_transpose_1/strided_slice_1/stack_1:output:0Vbigger_auto_encoder/bigger_decoder/conv2d_transpose_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2G
Ebigger_auto_encoder/bigger_decoder/conv2d_transpose_1/strided_slice_1�
Ubigger_auto_encoder/bigger_decoder/conv2d_transpose_1/conv2d_transpose/ReadVariableOpReadVariableOp^bigger_auto_encoder_bigger_decoder_conv2d_transpose_1_conv2d_transpose_readvariableop_resource*&
_output_shapes
:@@*
dtype02W
Ubigger_auto_encoder/bigger_decoder/conv2d_transpose_1/conv2d_transpose/ReadVariableOp�
Fbigger_auto_encoder/bigger_decoder/conv2d_transpose_1/conv2d_transposeConv2DBackpropInputDbigger_auto_encoder/bigger_decoder/conv2d_transpose_1/stack:output:0]bigger_auto_encoder/bigger_decoder/conv2d_transpose_1/conv2d_transpose/ReadVariableOp:value:0Fbigger_auto_encoder/bigger_decoder/conv2d_transpose/Relu:activations:0*
T0*1
_output_shapes
:�����������@*
paddingSAME*
strides
2H
Fbigger_auto_encoder/bigger_decoder/conv2d_transpose_1/conv2d_transpose�
Lbigger_auto_encoder/bigger_decoder/conv2d_transpose_1/BiasAdd/ReadVariableOpReadVariableOpUbigger_auto_encoder_bigger_decoder_conv2d_transpose_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02N
Lbigger_auto_encoder/bigger_decoder/conv2d_transpose_1/BiasAdd/ReadVariableOp�
=bigger_auto_encoder/bigger_decoder/conv2d_transpose_1/BiasAddBiasAddObigger_auto_encoder/bigger_decoder/conv2d_transpose_1/conv2d_transpose:output:0Tbigger_auto_encoder/bigger_decoder/conv2d_transpose_1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������@2?
=bigger_auto_encoder/bigger_decoder/conv2d_transpose_1/BiasAdd�
:bigger_auto_encoder/bigger_decoder/conv2d_transpose_1/ReluReluFbigger_auto_encoder/bigger_decoder/conv2d_transpose_1/BiasAdd:output:0*
T0*1
_output_shapes
:�����������@2<
:bigger_auto_encoder/bigger_decoder/conv2d_transpose_1/Relu�
;bigger_auto_encoder/bigger_decoder/conv2d_transpose_2/ShapeShapeHbigger_auto_encoder/bigger_decoder/conv2d_transpose_1/Relu:activations:0*
T0*
_output_shapes
:2=
;bigger_auto_encoder/bigger_decoder/conv2d_transpose_2/Shape�
Ibigger_auto_encoder/bigger_decoder/conv2d_transpose_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2K
Ibigger_auto_encoder/bigger_decoder/conv2d_transpose_2/strided_slice/stack�
Kbigger_auto_encoder/bigger_decoder/conv2d_transpose_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2M
Kbigger_auto_encoder/bigger_decoder/conv2d_transpose_2/strided_slice/stack_1�
Kbigger_auto_encoder/bigger_decoder/conv2d_transpose_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2M
Kbigger_auto_encoder/bigger_decoder/conv2d_transpose_2/strided_slice/stack_2�
Cbigger_auto_encoder/bigger_decoder/conv2d_transpose_2/strided_sliceStridedSliceDbigger_auto_encoder/bigger_decoder/conv2d_transpose_2/Shape:output:0Rbigger_auto_encoder/bigger_decoder/conv2d_transpose_2/strided_slice/stack:output:0Tbigger_auto_encoder/bigger_decoder/conv2d_transpose_2/strided_slice/stack_1:output:0Tbigger_auto_encoder/bigger_decoder/conv2d_transpose_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2E
Cbigger_auto_encoder/bigger_decoder/conv2d_transpose_2/strided_slice�
=bigger_auto_encoder/bigger_decoder/conv2d_transpose_2/stack/1Const*
_output_shapes
: *
dtype0*
value
B :�2?
=bigger_auto_encoder/bigger_decoder/conv2d_transpose_2/stack/1�
=bigger_auto_encoder/bigger_decoder/conv2d_transpose_2/stack/2Const*
_output_shapes
: *
dtype0*
value
B :�2?
=bigger_auto_encoder/bigger_decoder/conv2d_transpose_2/stack/2�
=bigger_auto_encoder/bigger_decoder/conv2d_transpose_2/stack/3Const*
_output_shapes
: *
dtype0*
value	B : 2?
=bigger_auto_encoder/bigger_decoder/conv2d_transpose_2/stack/3�
;bigger_auto_encoder/bigger_decoder/conv2d_transpose_2/stackPackLbigger_auto_encoder/bigger_decoder/conv2d_transpose_2/strided_slice:output:0Fbigger_auto_encoder/bigger_decoder/conv2d_transpose_2/stack/1:output:0Fbigger_auto_encoder/bigger_decoder/conv2d_transpose_2/stack/2:output:0Fbigger_auto_encoder/bigger_decoder/conv2d_transpose_2/stack/3:output:0*
N*
T0*
_output_shapes
:2=
;bigger_auto_encoder/bigger_decoder/conv2d_transpose_2/stack�
Kbigger_auto_encoder/bigger_decoder/conv2d_transpose_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2M
Kbigger_auto_encoder/bigger_decoder/conv2d_transpose_2/strided_slice_1/stack�
Mbigger_auto_encoder/bigger_decoder/conv2d_transpose_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2O
Mbigger_auto_encoder/bigger_decoder/conv2d_transpose_2/strided_slice_1/stack_1�
Mbigger_auto_encoder/bigger_decoder/conv2d_transpose_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2O
Mbigger_auto_encoder/bigger_decoder/conv2d_transpose_2/strided_slice_1/stack_2�
Ebigger_auto_encoder/bigger_decoder/conv2d_transpose_2/strided_slice_1StridedSliceDbigger_auto_encoder/bigger_decoder/conv2d_transpose_2/stack:output:0Tbigger_auto_encoder/bigger_decoder/conv2d_transpose_2/strided_slice_1/stack:output:0Vbigger_auto_encoder/bigger_decoder/conv2d_transpose_2/strided_slice_1/stack_1:output:0Vbigger_auto_encoder/bigger_decoder/conv2d_transpose_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2G
Ebigger_auto_encoder/bigger_decoder/conv2d_transpose_2/strided_slice_1�
Ubigger_auto_encoder/bigger_decoder/conv2d_transpose_2/conv2d_transpose/ReadVariableOpReadVariableOp^bigger_auto_encoder_bigger_decoder_conv2d_transpose_2_conv2d_transpose_readvariableop_resource*&
_output_shapes
: @*
dtype02W
Ubigger_auto_encoder/bigger_decoder/conv2d_transpose_2/conv2d_transpose/ReadVariableOp�
Fbigger_auto_encoder/bigger_decoder/conv2d_transpose_2/conv2d_transposeConv2DBackpropInputDbigger_auto_encoder/bigger_decoder/conv2d_transpose_2/stack:output:0]bigger_auto_encoder/bigger_decoder/conv2d_transpose_2/conv2d_transpose/ReadVariableOp:value:0Hbigger_auto_encoder/bigger_decoder/conv2d_transpose_1/Relu:activations:0*
T0*1
_output_shapes
:����������� *
paddingSAME*
strides
2H
Fbigger_auto_encoder/bigger_decoder/conv2d_transpose_2/conv2d_transpose�
Lbigger_auto_encoder/bigger_decoder/conv2d_transpose_2/BiasAdd/ReadVariableOpReadVariableOpUbigger_auto_encoder_bigger_decoder_conv2d_transpose_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02N
Lbigger_auto_encoder/bigger_decoder/conv2d_transpose_2/BiasAdd/ReadVariableOp�
=bigger_auto_encoder/bigger_decoder/conv2d_transpose_2/BiasAddBiasAddObigger_auto_encoder/bigger_decoder/conv2d_transpose_2/conv2d_transpose:output:0Tbigger_auto_encoder/bigger_decoder/conv2d_transpose_2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:����������� 2?
=bigger_auto_encoder/bigger_decoder/conv2d_transpose_2/BiasAdd�
:bigger_auto_encoder/bigger_decoder/conv2d_transpose_2/ReluReluFbigger_auto_encoder/bigger_decoder/conv2d_transpose_2/BiasAdd:output:0*
T0*1
_output_shapes
:����������� 2<
:bigger_auto_encoder/bigger_decoder/conv2d_transpose_2/Relu�
;bigger_auto_encoder/bigger_decoder/conv2d_transpose_3/ShapeShapeHbigger_auto_encoder/bigger_decoder/conv2d_transpose_2/Relu:activations:0*
T0*
_output_shapes
:2=
;bigger_auto_encoder/bigger_decoder/conv2d_transpose_3/Shape�
Ibigger_auto_encoder/bigger_decoder/conv2d_transpose_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2K
Ibigger_auto_encoder/bigger_decoder/conv2d_transpose_3/strided_slice/stack�
Kbigger_auto_encoder/bigger_decoder/conv2d_transpose_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2M
Kbigger_auto_encoder/bigger_decoder/conv2d_transpose_3/strided_slice/stack_1�
Kbigger_auto_encoder/bigger_decoder/conv2d_transpose_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2M
Kbigger_auto_encoder/bigger_decoder/conv2d_transpose_3/strided_slice/stack_2�
Cbigger_auto_encoder/bigger_decoder/conv2d_transpose_3/strided_sliceStridedSliceDbigger_auto_encoder/bigger_decoder/conv2d_transpose_3/Shape:output:0Rbigger_auto_encoder/bigger_decoder/conv2d_transpose_3/strided_slice/stack:output:0Tbigger_auto_encoder/bigger_decoder/conv2d_transpose_3/strided_slice/stack_1:output:0Tbigger_auto_encoder/bigger_decoder/conv2d_transpose_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2E
Cbigger_auto_encoder/bigger_decoder/conv2d_transpose_3/strided_slice�
=bigger_auto_encoder/bigger_decoder/conv2d_transpose_3/stack/1Const*
_output_shapes
: *
dtype0*
value
B :�2?
=bigger_auto_encoder/bigger_decoder/conv2d_transpose_3/stack/1�
=bigger_auto_encoder/bigger_decoder/conv2d_transpose_3/stack/2Const*
_output_shapes
: *
dtype0*
value
B :�2?
=bigger_auto_encoder/bigger_decoder/conv2d_transpose_3/stack/2�
=bigger_auto_encoder/bigger_decoder/conv2d_transpose_3/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2?
=bigger_auto_encoder/bigger_decoder/conv2d_transpose_3/stack/3�
;bigger_auto_encoder/bigger_decoder/conv2d_transpose_3/stackPackLbigger_auto_encoder/bigger_decoder/conv2d_transpose_3/strided_slice:output:0Fbigger_auto_encoder/bigger_decoder/conv2d_transpose_3/stack/1:output:0Fbigger_auto_encoder/bigger_decoder/conv2d_transpose_3/stack/2:output:0Fbigger_auto_encoder/bigger_decoder/conv2d_transpose_3/stack/3:output:0*
N*
T0*
_output_shapes
:2=
;bigger_auto_encoder/bigger_decoder/conv2d_transpose_3/stack�
Kbigger_auto_encoder/bigger_decoder/conv2d_transpose_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2M
Kbigger_auto_encoder/bigger_decoder/conv2d_transpose_3/strided_slice_1/stack�
Mbigger_auto_encoder/bigger_decoder/conv2d_transpose_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2O
Mbigger_auto_encoder/bigger_decoder/conv2d_transpose_3/strided_slice_1/stack_1�
Mbigger_auto_encoder/bigger_decoder/conv2d_transpose_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2O
Mbigger_auto_encoder/bigger_decoder/conv2d_transpose_3/strided_slice_1/stack_2�
Ebigger_auto_encoder/bigger_decoder/conv2d_transpose_3/strided_slice_1StridedSliceDbigger_auto_encoder/bigger_decoder/conv2d_transpose_3/stack:output:0Tbigger_auto_encoder/bigger_decoder/conv2d_transpose_3/strided_slice_1/stack:output:0Vbigger_auto_encoder/bigger_decoder/conv2d_transpose_3/strided_slice_1/stack_1:output:0Vbigger_auto_encoder/bigger_decoder/conv2d_transpose_3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2G
Ebigger_auto_encoder/bigger_decoder/conv2d_transpose_3/strided_slice_1�
Ubigger_auto_encoder/bigger_decoder/conv2d_transpose_3/conv2d_transpose/ReadVariableOpReadVariableOp^bigger_auto_encoder_bigger_decoder_conv2d_transpose_3_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype02W
Ubigger_auto_encoder/bigger_decoder/conv2d_transpose_3/conv2d_transpose/ReadVariableOp�
Fbigger_auto_encoder/bigger_decoder/conv2d_transpose_3/conv2d_transposeConv2DBackpropInputDbigger_auto_encoder/bigger_decoder/conv2d_transpose_3/stack:output:0]bigger_auto_encoder/bigger_decoder/conv2d_transpose_3/conv2d_transpose/ReadVariableOp:value:0Hbigger_auto_encoder/bigger_decoder/conv2d_transpose_2/Relu:activations:0*
T0*1
_output_shapes
:�����������*
paddingSAME*
strides
2H
Fbigger_auto_encoder/bigger_decoder/conv2d_transpose_3/conv2d_transpose�
Lbigger_auto_encoder/bigger_decoder/conv2d_transpose_3/BiasAdd/ReadVariableOpReadVariableOpUbigger_auto_encoder_bigger_decoder_conv2d_transpose_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02N
Lbigger_auto_encoder/bigger_decoder/conv2d_transpose_3/BiasAdd/ReadVariableOp�
=bigger_auto_encoder/bigger_decoder/conv2d_transpose_3/BiasAddBiasAddObigger_auto_encoder/bigger_decoder/conv2d_transpose_3/conv2d_transpose:output:0Tbigger_auto_encoder/bigger_decoder/conv2d_transpose_3/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������2?
=bigger_auto_encoder/bigger_decoder/conv2d_transpose_3/BiasAdd�
:bigger_auto_encoder/bigger_decoder/conv2d_transpose_3/ReluReluFbigger_auto_encoder/bigger_decoder/conv2d_transpose_3/BiasAdd:output:0*
T0*1
_output_shapes
:�����������2<
:bigger_auto_encoder/bigger_decoder/conv2d_transpose_3/Relu�
6bigger_auto_encoder/bigger_decoder/up_sampling2d/ShapeShapeHbigger_auto_encoder/bigger_decoder/conv2d_transpose_3/Relu:activations:0*
T0*
_output_shapes
:28
6bigger_auto_encoder/bigger_decoder/up_sampling2d/Shape�
Dbigger_auto_encoder/bigger_decoder/up_sampling2d/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2F
Dbigger_auto_encoder/bigger_decoder/up_sampling2d/strided_slice/stack�
Fbigger_auto_encoder/bigger_decoder/up_sampling2d/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2H
Fbigger_auto_encoder/bigger_decoder/up_sampling2d/strided_slice/stack_1�
Fbigger_auto_encoder/bigger_decoder/up_sampling2d/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2H
Fbigger_auto_encoder/bigger_decoder/up_sampling2d/strided_slice/stack_2�
>bigger_auto_encoder/bigger_decoder/up_sampling2d/strided_sliceStridedSlice?bigger_auto_encoder/bigger_decoder/up_sampling2d/Shape:output:0Mbigger_auto_encoder/bigger_decoder/up_sampling2d/strided_slice/stack:output:0Obigger_auto_encoder/bigger_decoder/up_sampling2d/strided_slice/stack_1:output:0Obigger_auto_encoder/bigger_decoder/up_sampling2d/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2@
>bigger_auto_encoder/bigger_decoder/up_sampling2d/strided_slice�
6bigger_auto_encoder/bigger_decoder/up_sampling2d/ConstConst*
_output_shapes
:*
dtype0*
valueB"      28
6bigger_auto_encoder/bigger_decoder/up_sampling2d/Const�
4bigger_auto_encoder/bigger_decoder/up_sampling2d/mulMulGbigger_auto_encoder/bigger_decoder/up_sampling2d/strided_slice:output:0?bigger_auto_encoder/bigger_decoder/up_sampling2d/Const:output:0*
T0*
_output_shapes
:26
4bigger_auto_encoder/bigger_decoder/up_sampling2d/mul�
Mbigger_auto_encoder/bigger_decoder/up_sampling2d/resize/ResizeNearestNeighborResizeNearestNeighborHbigger_auto_encoder/bigger_decoder/conv2d_transpose_3/Relu:activations:08bigger_auto_encoder/bigger_decoder/up_sampling2d/mul:z:0*
T0*1
_output_shapes
:�����������*
half_pixel_centers(2O
Mbigger_auto_encoder/bigger_decoder/up_sampling2d/resize/ResizeNearestNeighbor�
Abigger_auto_encoder/bigger_decoder/conv2d_4/Conv2D/ReadVariableOpReadVariableOpJbigger_auto_encoder_bigger_decoder_conv2d_4_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02C
Abigger_auto_encoder/bigger_decoder/conv2d_4/Conv2D/ReadVariableOp�
2bigger_auto_encoder/bigger_decoder/conv2d_4/Conv2DConv2D^bigger_auto_encoder/bigger_decoder/up_sampling2d/resize/ResizeNearestNeighbor:resized_images:0Ibigger_auto_encoder/bigger_decoder/conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������*
paddingSAME*
strides
24
2bigger_auto_encoder/bigger_decoder/conv2d_4/Conv2D�
Bbigger_auto_encoder/bigger_decoder/conv2d_4/BiasAdd/ReadVariableOpReadVariableOpKbigger_auto_encoder_bigger_decoder_conv2d_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02D
Bbigger_auto_encoder/bigger_decoder/conv2d_4/BiasAdd/ReadVariableOp�
3bigger_auto_encoder/bigger_decoder/conv2d_4/BiasAddBiasAdd;bigger_auto_encoder/bigger_decoder/conv2d_4/Conv2D:output:0Jbigger_auto_encoder/bigger_decoder/conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������25
3bigger_auto_encoder/bigger_decoder/conv2d_4/BiasAdd�
0bigger_auto_encoder/bigger_decoder/conv2d_4/TanhTanh<bigger_auto_encoder/bigger_decoder/conv2d_4/BiasAdd:output:0*
T0*1
_output_shapes
:�����������22
0bigger_auto_encoder/bigger_decoder/conv2d_4/Tanh�
IdentityIdentity4bigger_auto_encoder/bigger_decoder/conv2d_4/Tanh:y:0C^bigger_auto_encoder/bigger_decoder/conv2d_4/BiasAdd/ReadVariableOpB^bigger_auto_encoder/bigger_decoder/conv2d_4/Conv2D/ReadVariableOpK^bigger_auto_encoder/bigger_decoder/conv2d_transpose/BiasAdd/ReadVariableOpT^bigger_auto_encoder/bigger_decoder/conv2d_transpose/conv2d_transpose/ReadVariableOpM^bigger_auto_encoder/bigger_decoder/conv2d_transpose_1/BiasAdd/ReadVariableOpV^bigger_auto_encoder/bigger_decoder/conv2d_transpose_1/conv2d_transpose/ReadVariableOpM^bigger_auto_encoder/bigger_decoder/conv2d_transpose_2/BiasAdd/ReadVariableOpV^bigger_auto_encoder/bigger_decoder/conv2d_transpose_2/conv2d_transpose/ReadVariableOpM^bigger_auto_encoder/bigger_decoder/conv2d_transpose_3/BiasAdd/ReadVariableOpV^bigger_auto_encoder/bigger_decoder/conv2d_transpose_3/conv2d_transpose/ReadVariableOpA^bigger_auto_encoder/bigger_encoder/conv2d/BiasAdd/ReadVariableOp@^bigger_auto_encoder/bigger_encoder/conv2d/Conv2D/ReadVariableOpC^bigger_auto_encoder/bigger_encoder/conv2d_1/BiasAdd/ReadVariableOpB^bigger_auto_encoder/bigger_encoder/conv2d_1/Conv2D/ReadVariableOpC^bigger_auto_encoder/bigger_encoder/conv2d_2/BiasAdd/ReadVariableOpB^bigger_auto_encoder/bigger_encoder/conv2d_2/Conv2D/ReadVariableOpC^bigger_auto_encoder/bigger_encoder/conv2d_3/BiasAdd/ReadVariableOpB^bigger_auto_encoder/bigger_encoder/conv2d_3/Conv2D/ReadVariableOp*
T0*1
_output_shapes
:�����������2

Identity"
identityIdentity:output:0*x
_input_shapesg
e:�����������::::::::::::::::::2�
Bbigger_auto_encoder/bigger_decoder/conv2d_4/BiasAdd/ReadVariableOpBbigger_auto_encoder/bigger_decoder/conv2d_4/BiasAdd/ReadVariableOp2�
Abigger_auto_encoder/bigger_decoder/conv2d_4/Conv2D/ReadVariableOpAbigger_auto_encoder/bigger_decoder/conv2d_4/Conv2D/ReadVariableOp2�
Jbigger_auto_encoder/bigger_decoder/conv2d_transpose/BiasAdd/ReadVariableOpJbigger_auto_encoder/bigger_decoder/conv2d_transpose/BiasAdd/ReadVariableOp2�
Sbigger_auto_encoder/bigger_decoder/conv2d_transpose/conv2d_transpose/ReadVariableOpSbigger_auto_encoder/bigger_decoder/conv2d_transpose/conv2d_transpose/ReadVariableOp2�
Lbigger_auto_encoder/bigger_decoder/conv2d_transpose_1/BiasAdd/ReadVariableOpLbigger_auto_encoder/bigger_decoder/conv2d_transpose_1/BiasAdd/ReadVariableOp2�
Ubigger_auto_encoder/bigger_decoder/conv2d_transpose_1/conv2d_transpose/ReadVariableOpUbigger_auto_encoder/bigger_decoder/conv2d_transpose_1/conv2d_transpose/ReadVariableOp2�
Lbigger_auto_encoder/bigger_decoder/conv2d_transpose_2/BiasAdd/ReadVariableOpLbigger_auto_encoder/bigger_decoder/conv2d_transpose_2/BiasAdd/ReadVariableOp2�
Ubigger_auto_encoder/bigger_decoder/conv2d_transpose_2/conv2d_transpose/ReadVariableOpUbigger_auto_encoder/bigger_decoder/conv2d_transpose_2/conv2d_transpose/ReadVariableOp2�
Lbigger_auto_encoder/bigger_decoder/conv2d_transpose_3/BiasAdd/ReadVariableOpLbigger_auto_encoder/bigger_decoder/conv2d_transpose_3/BiasAdd/ReadVariableOp2�
Ubigger_auto_encoder/bigger_decoder/conv2d_transpose_3/conv2d_transpose/ReadVariableOpUbigger_auto_encoder/bigger_decoder/conv2d_transpose_3/conv2d_transpose/ReadVariableOp2�
@bigger_auto_encoder/bigger_encoder/conv2d/BiasAdd/ReadVariableOp@bigger_auto_encoder/bigger_encoder/conv2d/BiasAdd/ReadVariableOp2�
?bigger_auto_encoder/bigger_encoder/conv2d/Conv2D/ReadVariableOp?bigger_auto_encoder/bigger_encoder/conv2d/Conv2D/ReadVariableOp2�
Bbigger_auto_encoder/bigger_encoder/conv2d_1/BiasAdd/ReadVariableOpBbigger_auto_encoder/bigger_encoder/conv2d_1/BiasAdd/ReadVariableOp2�
Abigger_auto_encoder/bigger_encoder/conv2d_1/Conv2D/ReadVariableOpAbigger_auto_encoder/bigger_encoder/conv2d_1/Conv2D/ReadVariableOp2�
Bbigger_auto_encoder/bigger_encoder/conv2d_2/BiasAdd/ReadVariableOpBbigger_auto_encoder/bigger_encoder/conv2d_2/BiasAdd/ReadVariableOp2�
Abigger_auto_encoder/bigger_encoder/conv2d_2/Conv2D/ReadVariableOpAbigger_auto_encoder/bigger_encoder/conv2d_2/Conv2D/ReadVariableOp2�
Bbigger_auto_encoder/bigger_encoder/conv2d_3/BiasAdd/ReadVariableOpBbigger_auto_encoder/bigger_encoder/conv2d_3/BiasAdd/ReadVariableOp2�
Abigger_auto_encoder/bigger_encoder/conv2d_3/Conv2D/ReadVariableOpAbigger_auto_encoder/bigger_encoder/conv2d_3/Conv2D/ReadVariableOp:Z V
1
_output_shapes
:�����������
!
_user_specified_name	input_1
�
�
O__inference_bigger_auto_encoder_layer_call_and_return_conditional_losses_144182
input_1
bigger_encoder_144143
bigger_encoder_144145
bigger_encoder_144147
bigger_encoder_144149
bigger_encoder_144151
bigger_encoder_144153
bigger_encoder_144155
bigger_encoder_144157
bigger_decoder_144160
bigger_decoder_144162
bigger_decoder_144164
bigger_decoder_144166
bigger_decoder_144168
bigger_decoder_144170
bigger_decoder_144172
bigger_decoder_144174
bigger_decoder_144176
bigger_decoder_144178
identity��&bigger_decoder/StatefulPartitionedCall�&bigger_encoder/StatefulPartitionedCall�
&bigger_encoder/StatefulPartitionedCallStatefulPartitionedCallinput_1bigger_encoder_144143bigger_encoder_144145bigger_encoder_144147bigger_encoder_144149bigger_encoder_144151bigger_encoder_144153bigger_encoder_144155bigger_encoder_144157*
Tin
2	*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������@**
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8� *S
fNRL
J__inference_bigger_encoder_layer_call_and_return_conditional_losses_1438392(
&bigger_encoder/StatefulPartitionedCall�
&bigger_decoder/StatefulPartitionedCallStatefulPartitionedCall/bigger_encoder/StatefulPartitionedCall:output:0bigger_decoder_144160bigger_decoder_144162bigger_decoder_144164bigger_decoder_144166bigger_decoder_144168bigger_decoder_144170bigger_decoder_144172bigger_decoder_144174bigger_decoder_144176bigger_decoder_144178*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*,
_read_only_resource_inputs

	
*2
config_proto" 

CPU

GPU2 *0J 8� *S
fNRL
J__inference_bigger_decoder_layer_call_and_return_conditional_losses_1441132(
&bigger_decoder/StatefulPartitionedCall�
IdentityIdentity/bigger_decoder/StatefulPartitionedCall:output:0'^bigger_decoder/StatefulPartitionedCall'^bigger_encoder/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*x
_input_shapesg
e:�����������::::::::::::::::::2P
&bigger_decoder/StatefulPartitionedCall&bigger_decoder/StatefulPartitionedCall2P
&bigger_encoder/StatefulPartitionedCall&bigger_encoder/StatefulPartitionedCall:Z V
1
_output_shapes
:�����������
!
_user_specified_name	input_1
�$
�
L__inference_conv2d_transpose_layer_call_and_return_conditional_losses_143896

inputs,
(conv2d_transpose_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�conv2d_transpose/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2�
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1x
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2�
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2P
mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
mul/y\
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mulT
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
mul_1/yb
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: 2
mul_1T
stack/3Const*
_output_shapes
: *
dtype0*
value	B :@2	
stack/3�
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:2
stackx
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2�
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3�
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:@@*
dtype02!
conv2d_transpose/ReadVariableOp�
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+���������������������������@*
paddingSAME*
strides
2
conv2d_transpose�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+���������������������������@2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+���������������������������@2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*
T0*A
_output_shapes/
-:+���������������������������@2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+���������������������������@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+���������������������������@
 
_user_specified_nameinputs
�	
�
/__inference_bigger_decoder_layer_call_fn_144139
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*,
_read_only_resource_inputs

	
*2
config_proto" 

CPU

GPU2 *0J 8� *S
fNRL
J__inference_bigger_decoder_layer_call_and_return_conditional_losses_1441132
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*X
_input_shapesG
E:�����������@::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:�����������@
!
_user_specified_name	input_1
�$
�
N__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_143941

inputs,
(conv2d_transpose_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�conv2d_transpose/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2�
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1x
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2�
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2P
mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
mul/y\
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mulT
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
mul_1/yb
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: 2
mul_1T
stack/3Const*
_output_shapes
: *
dtype0*
value	B :@2	
stack/3�
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:2
stackx
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2�
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3�
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:@@*
dtype02!
conv2d_transpose/ReadVariableOp�
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+���������������������������@*
paddingSAME*
strides
2
conv2d_transpose�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+���������������������������@2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+���������������������������@2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*
T0*A
_output_shapes/
-:+���������������������������@2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+���������������������������@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+���������������������������@
 
_user_specified_nameinputs
�
�
4__inference_bigger_auto_encoder_layer_call_fn_144224
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*4
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8� *X
fSRQ
O__inference_bigger_auto_encoder_layer_call_and_return_conditional_losses_1441822
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*x
_input_shapesg
e:�����������::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:�����������
!
_user_specified_name	input_1
�
J
.__inference_up_sampling2d_layer_call_fn_144060

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4������������������������������������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *R
fMRK
I__inference_up_sampling2d_layer_call_and_return_conditional_losses_1440542
PartitionedCall�
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4������������������������������������2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
e
I__inference_up_sampling2d_layer_call_and_return_conditional_losses_144054

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
strided_slice_
ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
Const^
mulMulstrided_slice:output:0Const:output:0*
T0*
_output_shapes
:2
mul�
resize/ResizeNearestNeighborResizeNearestNeighborinputsmul:z:0*
T0*J
_output_shapes8
6:4������������������������������������*
half_pixel_centers(2
resize/ResizeNearestNeighbor�
IdentityIdentity-resize/ResizeNearestNeighbor:resized_images:0*
T0*J
_output_shapes8
6:4������������������������������������2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�

�
D__inference_conv2d_3_layer_call_and_return_conditional_losses_144346

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������@*
paddingSAME*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������@2	
BiasAddb
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:�����������@2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:�����������@2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:�����������@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:�����������@
 
_user_specified_nameinputs
�
�
D__inference_conv2d_4_layer_call_and_return_conditional_losses_144366

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+���������������������������*
paddingSAME*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+���������������������������2	
BiasAddr
TanhTanhBiasAdd:output:0*
T0*A
_output_shapes/
-:+���������������������������2
Tanh�
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+���������������������������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�
�
3__inference_conv2d_transpose_3_layer_call_fn_144041

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *W
fRRP
N__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_1440312
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+��������������������������� ::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+��������������������������� 
 
_user_specified_nameinputs
��
�*
__inference__traced_save_144587
file_prefix(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableopO
Ksavev2_bigger_auto_encoder_bigger_encoder_conv2d_kernel_read_readvariableopM
Isavev2_bigger_auto_encoder_bigger_encoder_conv2d_bias_read_readvariableopQ
Msavev2_bigger_auto_encoder_bigger_encoder_conv2d_1_kernel_read_readvariableopO
Ksavev2_bigger_auto_encoder_bigger_encoder_conv2d_1_bias_read_readvariableopQ
Msavev2_bigger_auto_encoder_bigger_encoder_conv2d_2_kernel_read_readvariableopO
Ksavev2_bigger_auto_encoder_bigger_encoder_conv2d_2_bias_read_readvariableopQ
Msavev2_bigger_auto_encoder_bigger_encoder_conv2d_3_kernel_read_readvariableopO
Ksavev2_bigger_auto_encoder_bigger_encoder_conv2d_3_bias_read_readvariableopY
Usavev2_bigger_auto_encoder_bigger_decoder_conv2d_transpose_kernel_read_readvariableopW
Ssavev2_bigger_auto_encoder_bigger_decoder_conv2d_transpose_bias_read_readvariableop[
Wsavev2_bigger_auto_encoder_bigger_decoder_conv2d_transpose_1_kernel_read_readvariableopY
Usavev2_bigger_auto_encoder_bigger_decoder_conv2d_transpose_1_bias_read_readvariableop[
Wsavev2_bigger_auto_encoder_bigger_decoder_conv2d_transpose_2_kernel_read_readvariableopY
Usavev2_bigger_auto_encoder_bigger_decoder_conv2d_transpose_2_bias_read_readvariableop[
Wsavev2_bigger_auto_encoder_bigger_decoder_conv2d_transpose_3_kernel_read_readvariableopY
Usavev2_bigger_auto_encoder_bigger_decoder_conv2d_transpose_3_bias_read_readvariableopQ
Msavev2_bigger_auto_encoder_bigger_decoder_conv2d_4_kernel_read_readvariableopO
Ksavev2_bigger_auto_encoder_bigger_decoder_conv2d_4_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableopV
Rsavev2_adam_bigger_auto_encoder_bigger_encoder_conv2d_kernel_m_read_readvariableopT
Psavev2_adam_bigger_auto_encoder_bigger_encoder_conv2d_bias_m_read_readvariableopX
Tsavev2_adam_bigger_auto_encoder_bigger_encoder_conv2d_1_kernel_m_read_readvariableopV
Rsavev2_adam_bigger_auto_encoder_bigger_encoder_conv2d_1_bias_m_read_readvariableopX
Tsavev2_adam_bigger_auto_encoder_bigger_encoder_conv2d_2_kernel_m_read_readvariableopV
Rsavev2_adam_bigger_auto_encoder_bigger_encoder_conv2d_2_bias_m_read_readvariableopX
Tsavev2_adam_bigger_auto_encoder_bigger_encoder_conv2d_3_kernel_m_read_readvariableopV
Rsavev2_adam_bigger_auto_encoder_bigger_encoder_conv2d_3_bias_m_read_readvariableop`
\savev2_adam_bigger_auto_encoder_bigger_decoder_conv2d_transpose_kernel_m_read_readvariableop^
Zsavev2_adam_bigger_auto_encoder_bigger_decoder_conv2d_transpose_bias_m_read_readvariableopb
^savev2_adam_bigger_auto_encoder_bigger_decoder_conv2d_transpose_1_kernel_m_read_readvariableop`
\savev2_adam_bigger_auto_encoder_bigger_decoder_conv2d_transpose_1_bias_m_read_readvariableopb
^savev2_adam_bigger_auto_encoder_bigger_decoder_conv2d_transpose_2_kernel_m_read_readvariableop`
\savev2_adam_bigger_auto_encoder_bigger_decoder_conv2d_transpose_2_bias_m_read_readvariableopb
^savev2_adam_bigger_auto_encoder_bigger_decoder_conv2d_transpose_3_kernel_m_read_readvariableop`
\savev2_adam_bigger_auto_encoder_bigger_decoder_conv2d_transpose_3_bias_m_read_readvariableopX
Tsavev2_adam_bigger_auto_encoder_bigger_decoder_conv2d_4_kernel_m_read_readvariableopV
Rsavev2_adam_bigger_auto_encoder_bigger_decoder_conv2d_4_bias_m_read_readvariableopV
Rsavev2_adam_bigger_auto_encoder_bigger_encoder_conv2d_kernel_v_read_readvariableopT
Psavev2_adam_bigger_auto_encoder_bigger_encoder_conv2d_bias_v_read_readvariableopX
Tsavev2_adam_bigger_auto_encoder_bigger_encoder_conv2d_1_kernel_v_read_readvariableopV
Rsavev2_adam_bigger_auto_encoder_bigger_encoder_conv2d_1_bias_v_read_readvariableopX
Tsavev2_adam_bigger_auto_encoder_bigger_encoder_conv2d_2_kernel_v_read_readvariableopV
Rsavev2_adam_bigger_auto_encoder_bigger_encoder_conv2d_2_bias_v_read_readvariableopX
Tsavev2_adam_bigger_auto_encoder_bigger_encoder_conv2d_3_kernel_v_read_readvariableopV
Rsavev2_adam_bigger_auto_encoder_bigger_encoder_conv2d_3_bias_v_read_readvariableop`
\savev2_adam_bigger_auto_encoder_bigger_decoder_conv2d_transpose_kernel_v_read_readvariableop^
Zsavev2_adam_bigger_auto_encoder_bigger_decoder_conv2d_transpose_bias_v_read_readvariableopb
^savev2_adam_bigger_auto_encoder_bigger_decoder_conv2d_transpose_1_kernel_v_read_readvariableop`
\savev2_adam_bigger_auto_encoder_bigger_decoder_conv2d_transpose_1_bias_v_read_readvariableopb
^savev2_adam_bigger_auto_encoder_bigger_decoder_conv2d_transpose_2_kernel_v_read_readvariableop`
\savev2_adam_bigger_auto_encoder_bigger_decoder_conv2d_transpose_2_bias_v_read_readvariableopb
^savev2_adam_bigger_auto_encoder_bigger_decoder_conv2d_transpose_3_kernel_v_read_readvariableop`
\savev2_adam_bigger_auto_encoder_bigger_decoder_conv2d_transpose_3_bias_v_read_readvariableopX
Tsavev2_adam_bigger_auto_encoder_bigger_decoder_conv2d_4_kernel_v_read_readvariableopV
Rsavev2_adam_bigger_auto_encoder_bigger_decoder_conv2d_4_bias_v_read_readvariableop
savev2_const

identity_1��MergeV2Checkpoints�
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard�
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename�!
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:@*
dtype0*�!
value� B� @B)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/11/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/12/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/13/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/14/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/15/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/16/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/17/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:@*
dtype0*�
value�B�@B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices�)
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableopKsavev2_bigger_auto_encoder_bigger_encoder_conv2d_kernel_read_readvariableopIsavev2_bigger_auto_encoder_bigger_encoder_conv2d_bias_read_readvariableopMsavev2_bigger_auto_encoder_bigger_encoder_conv2d_1_kernel_read_readvariableopKsavev2_bigger_auto_encoder_bigger_encoder_conv2d_1_bias_read_readvariableopMsavev2_bigger_auto_encoder_bigger_encoder_conv2d_2_kernel_read_readvariableopKsavev2_bigger_auto_encoder_bigger_encoder_conv2d_2_bias_read_readvariableopMsavev2_bigger_auto_encoder_bigger_encoder_conv2d_3_kernel_read_readvariableopKsavev2_bigger_auto_encoder_bigger_encoder_conv2d_3_bias_read_readvariableopUsavev2_bigger_auto_encoder_bigger_decoder_conv2d_transpose_kernel_read_readvariableopSsavev2_bigger_auto_encoder_bigger_decoder_conv2d_transpose_bias_read_readvariableopWsavev2_bigger_auto_encoder_bigger_decoder_conv2d_transpose_1_kernel_read_readvariableopUsavev2_bigger_auto_encoder_bigger_decoder_conv2d_transpose_1_bias_read_readvariableopWsavev2_bigger_auto_encoder_bigger_decoder_conv2d_transpose_2_kernel_read_readvariableopUsavev2_bigger_auto_encoder_bigger_decoder_conv2d_transpose_2_bias_read_readvariableopWsavev2_bigger_auto_encoder_bigger_decoder_conv2d_transpose_3_kernel_read_readvariableopUsavev2_bigger_auto_encoder_bigger_decoder_conv2d_transpose_3_bias_read_readvariableopMsavev2_bigger_auto_encoder_bigger_decoder_conv2d_4_kernel_read_readvariableopKsavev2_bigger_auto_encoder_bigger_decoder_conv2d_4_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableopRsavev2_adam_bigger_auto_encoder_bigger_encoder_conv2d_kernel_m_read_readvariableopPsavev2_adam_bigger_auto_encoder_bigger_encoder_conv2d_bias_m_read_readvariableopTsavev2_adam_bigger_auto_encoder_bigger_encoder_conv2d_1_kernel_m_read_readvariableopRsavev2_adam_bigger_auto_encoder_bigger_encoder_conv2d_1_bias_m_read_readvariableopTsavev2_adam_bigger_auto_encoder_bigger_encoder_conv2d_2_kernel_m_read_readvariableopRsavev2_adam_bigger_auto_encoder_bigger_encoder_conv2d_2_bias_m_read_readvariableopTsavev2_adam_bigger_auto_encoder_bigger_encoder_conv2d_3_kernel_m_read_readvariableopRsavev2_adam_bigger_auto_encoder_bigger_encoder_conv2d_3_bias_m_read_readvariableop\savev2_adam_bigger_auto_encoder_bigger_decoder_conv2d_transpose_kernel_m_read_readvariableopZsavev2_adam_bigger_auto_encoder_bigger_decoder_conv2d_transpose_bias_m_read_readvariableop^savev2_adam_bigger_auto_encoder_bigger_decoder_conv2d_transpose_1_kernel_m_read_readvariableop\savev2_adam_bigger_auto_encoder_bigger_decoder_conv2d_transpose_1_bias_m_read_readvariableop^savev2_adam_bigger_auto_encoder_bigger_decoder_conv2d_transpose_2_kernel_m_read_readvariableop\savev2_adam_bigger_auto_encoder_bigger_decoder_conv2d_transpose_2_bias_m_read_readvariableop^savev2_adam_bigger_auto_encoder_bigger_decoder_conv2d_transpose_3_kernel_m_read_readvariableop\savev2_adam_bigger_auto_encoder_bigger_decoder_conv2d_transpose_3_bias_m_read_readvariableopTsavev2_adam_bigger_auto_encoder_bigger_decoder_conv2d_4_kernel_m_read_readvariableopRsavev2_adam_bigger_auto_encoder_bigger_decoder_conv2d_4_bias_m_read_readvariableopRsavev2_adam_bigger_auto_encoder_bigger_encoder_conv2d_kernel_v_read_readvariableopPsavev2_adam_bigger_auto_encoder_bigger_encoder_conv2d_bias_v_read_readvariableopTsavev2_adam_bigger_auto_encoder_bigger_encoder_conv2d_1_kernel_v_read_readvariableopRsavev2_adam_bigger_auto_encoder_bigger_encoder_conv2d_1_bias_v_read_readvariableopTsavev2_adam_bigger_auto_encoder_bigger_encoder_conv2d_2_kernel_v_read_readvariableopRsavev2_adam_bigger_auto_encoder_bigger_encoder_conv2d_2_bias_v_read_readvariableopTsavev2_adam_bigger_auto_encoder_bigger_encoder_conv2d_3_kernel_v_read_readvariableopRsavev2_adam_bigger_auto_encoder_bigger_encoder_conv2d_3_bias_v_read_readvariableop\savev2_adam_bigger_auto_encoder_bigger_decoder_conv2d_transpose_kernel_v_read_readvariableopZsavev2_adam_bigger_auto_encoder_bigger_decoder_conv2d_transpose_bias_v_read_readvariableop^savev2_adam_bigger_auto_encoder_bigger_decoder_conv2d_transpose_1_kernel_v_read_readvariableop\savev2_adam_bigger_auto_encoder_bigger_decoder_conv2d_transpose_1_bias_v_read_readvariableop^savev2_adam_bigger_auto_encoder_bigger_decoder_conv2d_transpose_2_kernel_v_read_readvariableop\savev2_adam_bigger_auto_encoder_bigger_decoder_conv2d_transpose_2_bias_v_read_readvariableop^savev2_adam_bigger_auto_encoder_bigger_decoder_conv2d_transpose_3_kernel_v_read_readvariableop\savev2_adam_bigger_auto_encoder_bigger_decoder_conv2d_transpose_3_bias_v_read_readvariableopTsavev2_adam_bigger_auto_encoder_bigger_decoder_conv2d_4_kernel_v_read_readvariableopRsavev2_adam_bigger_auto_encoder_bigger_decoder_conv2d_4_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *N
dtypesD
B2@	2
SaveV2�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*�
_input_shapes�
�: : : : : : ::: : : @:@:@@:@:@@:@:@@:@: @: : :::: : : : ::: : : @:@:@@:@:@@:@:@@:@: @: : :::::: : : @:@:@@:@:@@:@:@@:@: @: : :::: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
: : 	

_output_shapes
: :,
(
&
_output_shapes
: @: 

_output_shapes
:@:,(
&
_output_shapes
:@@: 

_output_shapes
:@:,(
&
_output_shapes
:@@: 

_output_shapes
:@:,(
&
_output_shapes
:@@: 

_output_shapes
:@:,(
&
_output_shapes
: @: 

_output_shapes
: :,(
&
_output_shapes
: : 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
: : 

_output_shapes
: :, (
&
_output_shapes
: @: !

_output_shapes
:@:,"(
&
_output_shapes
:@@: #

_output_shapes
:@:,$(
&
_output_shapes
:@@: %

_output_shapes
:@:,&(
&
_output_shapes
:@@: '

_output_shapes
:@:,((
&
_output_shapes
: @: )

_output_shapes
: :,*(
&
_output_shapes
: : +

_output_shapes
::,,(
&
_output_shapes
:: -

_output_shapes
::,.(
&
_output_shapes
:: /

_output_shapes
::,0(
&
_output_shapes
: : 1

_output_shapes
: :,2(
&
_output_shapes
: @: 3

_output_shapes
:@:,4(
&
_output_shapes
:@@: 5

_output_shapes
:@:,6(
&
_output_shapes
:@@: 7

_output_shapes
:@:,8(
&
_output_shapes
:@@: 9

_output_shapes
:@:,:(
&
_output_shapes
: @: ;

_output_shapes
: :,<(
&
_output_shapes
: : =

_output_shapes
::,>(
&
_output_shapes
:: ?

_output_shapes
::@

_output_shapes
: 
�

�
D__inference_conv2d_2_layer_call_and_return_conditional_losses_143795

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������@*
paddingSAME*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������@2	
BiasAddb
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:�����������@2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:�����������@2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:����������� ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:����������� 
 
_user_specified_nameinputs
�

�
D__inference_conv2d_1_layer_call_and_return_conditional_losses_143768

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:����������� *
paddingSAME*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:����������� 2	
BiasAddb
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:����������� 2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:����������� 2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:�����������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�
�
/__inference_bigger_encoder_layer_call_fn_143861
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������@**
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8� *S
fNRL
J__inference_bigger_encoder_layer_call_and_return_conditional_losses_1438392
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:�����������@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:�����������::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:�����������
!
_user_specified_name	input_1
�

�
D__inference_conv2d_1_layer_call_and_return_conditional_losses_144306

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:����������� *
paddingSAME*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:����������� 2	
BiasAddb
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:����������� 2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:����������� 2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:�����������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�
�
3__inference_conv2d_transpose_1_layer_call_fn_143951

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *W
fRRP
N__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_1439412
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������@2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+���������������������������@::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+���������������������������@
 
_user_specified_nameinputs
�
�
J__inference_bigger_encoder_layer_call_and_return_conditional_losses_143839
input_1
conv2d_143752
conv2d_143754
conv2d_1_143779
conv2d_1_143781
conv2d_2_143806
conv2d_2_143808
conv2d_3_143833
conv2d_3_143835
identity��conv2d/StatefulPartitionedCall� conv2d_1/StatefulPartitionedCall� conv2d_2/StatefulPartitionedCall� conv2d_3/StatefulPartitionedCall�
conv2d/StatefulPartitionedCallStatefulPartitionedCallinput_1conv2d_143752conv2d_143754*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *K
fFRD
B__inference_conv2d_layer_call_and_return_conditional_losses_1437412 
conv2d/StatefulPartitionedCall�
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0conv2d_1_143779conv2d_1_143781*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:����������� *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *M
fHRF
D__inference_conv2d_1_layer_call_and_return_conditional_losses_1437682"
 conv2d_1/StatefulPartitionedCall�
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0conv2d_2_143806conv2d_2_143808*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *M
fHRF
D__inference_conv2d_2_layer_call_and_return_conditional_losses_1437952"
 conv2d_2/StatefulPartitionedCall�
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0conv2d_3_143833conv2d_3_143835*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *M
fHRF
D__inference_conv2d_3_layer_call_and_return_conditional_losses_1438222"
 conv2d_3/StatefulPartitionedCall�
IdentityIdentity)conv2d_3/StatefulPartitionedCall:output:0^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall*
T0*1
_output_shapes
:�����������@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:�����������::::::::2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall:Z V
1
_output_shapes
:�����������
!
_user_specified_name	input_1
�

�
D__inference_conv2d_2_layer_call_and_return_conditional_losses_144326

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������@*
paddingSAME*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������@2	
BiasAddb
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:�����������@2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:�����������@2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:����������� ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:����������� 
 
_user_specified_nameinputs
�
~
)__inference_conv2d_1_layer_call_fn_144315

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:����������� *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *M
fHRF
D__inference_conv2d_1_layer_call_and_return_conditional_losses_1437682
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:����������� 2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:�����������::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs"�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
E
input_1:
serving_default_input_1:0�����������F
output_1:
StatefulPartitionedCall:0�����������tensorflow/serving/predict:��
�
encoder
decoder
	optimizer
regularization_losses
trainable_variables
	variables
	keras_api

signatures
�_default_save_signature
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_model�{"class_name": "BiggerAutoEncoder", "name": "bigger_auto_encoder", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"layer was saved without config": true}, "is_graph_network": false, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "BiggerAutoEncoder"}, "training_config": {"loss": "mse", "metrics": [[{"class_name": "MeanMetricWrapper", "config": {"name": "mae", "dtype": "float32", "fn": "mean_absolute_error"}}]], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 9.999999747378752e-05, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
�

	conv_1


conv_2

conv_3

conv_4
regularization_losses
trainable_variables
	variables
	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_model�{"class_name": "BiggerEncoder", "name": "bigger_encoder", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"layer was saved without config": true}, "is_graph_network": false, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "BiggerEncoder"}}
�

conv_1

conv_2

conv_3

conv_4
upsampling_4
conv_out
regularization_losses
trainable_variables
	variables
	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_model�{"class_name": "BiggerDecoder", "name": "bigger_decoder", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"layer was saved without config": true}, "is_graph_network": false, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "BiggerDecoder"}}
�
iter

beta_1

beta_2
	decay
learning_rate m�!m�"m�#m�$m�%m�&m�'m�(m�)m�*m�+m�,m�-m�.m�/m�0m�1m� v�!v�"v�#v�$v�%v�&v�'v�(v�)v�*v�+v�,v�-v�.v�/v�0v�1v�"
	optimizer
 "
trackable_list_wrapper
�
 0
!1
"2
#3
$4
%5
&6
'7
(8
)9
*10
+11
,12
-13
.14
/15
016
117"
trackable_list_wrapper
�
 0
!1
"2
#3
$4
%5
&6
'7
(8
)9
*10
+11
,12
-13
.14
/15
016
117"
trackable_list_wrapper
�
regularization_losses

2layers
3layer_metrics
4metrics
trainable_variables
5layer_regularization_losses
6non_trainable_variables
	variables
�__call__
�_default_save_signature
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
-
�serving_default"
signature_map
�	

 kernel
!bias
7regularization_losses
8trainable_variables
9	variables
:	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Conv2D", "name": "conv2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 1}}}, "build_input_shape": {"class_name": "TensorShape", "items": [30, 256, 256, 1]}}
�	

"kernel
#bias
;regularization_losses
<trainable_variables
=	variables
>	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Conv2D", "name": "conv2d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_1", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 16}}}, "build_input_shape": {"class_name": "TensorShape", "items": [30, 128, 128, 16]}}
�	

$kernel
%bias
?regularization_losses
@trainable_variables
A	variables
B	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Conv2D", "name": "conv2d_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [30, 128, 128, 32]}}
�	

&kernel
'bias
Cregularization_losses
Dtrainable_variables
E	variables
F	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Conv2D", "name": "conv2d_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_3", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [30, 128, 128, 64]}}
 "
trackable_list_wrapper
X
 0
!1
"2
#3
$4
%5
&6
'7"
trackable_list_wrapper
X
 0
!1
"2
#3
$4
%5
&6
'7"
trackable_list_wrapper
�
regularization_losses

Glayers
Hlayer_metrics
Imetrics
trainable_variables
Jlayer_regularization_losses
Knon_trainable_variables
	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�


(kernel
)bias
Lregularization_losses
Mtrainable_variables
N	variables
O	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Conv2DTranspose", "name": "conv2d_transpose", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_transpose", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [30, 128, 128, 64]}}
�


*kernel
+bias
Pregularization_losses
Qtrainable_variables
R	variables
S	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�	
_tf_keras_layer�{"class_name": "Conv2DTranspose", "name": "conv2d_transpose_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_transpose_1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [30, 128, 128, 64]}}
�


,kernel
-bias
Tregularization_losses
Utrainable_variables
V	variables
W	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�	
_tf_keras_layer�{"class_name": "Conv2DTranspose", "name": "conv2d_transpose_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_transpose_2", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [30, 128, 128, 64]}}
�


.kernel
/bias
Xregularization_losses
Ytrainable_variables
Z	variables
[	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�	
_tf_keras_layer�{"class_name": "Conv2DTranspose", "name": "conv2d_transpose_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_transpose_3", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [30, 128, 128, 32]}}
�
\regularization_losses
]trainable_variables
^	variables
_	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "UpSampling2D", "name": "up_sampling2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "up_sampling2d", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
�	

0kernel
1bias
`regularization_losses
atrainable_variables
b	variables
c	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Conv2D", "name": "conv2d_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_4", "trainable": true, "dtype": "float32", "filters": 3, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 16}}}, "build_input_shape": {"class_name": "TensorShape", "items": [30, 256, 256, 16]}}
 "
trackable_list_wrapper
f
(0
)1
*2
+3
,4
-5
.6
/7
08
19"
trackable_list_wrapper
f
(0
)1
*2
+3
,4
-5
.6
/7
08
19"
trackable_list_wrapper
�
regularization_losses

dlayers
elayer_metrics
fmetrics
trainable_variables
glayer_regularization_losses
hnon_trainable_variables
	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
J:H20bigger_auto_encoder/bigger_encoder/conv2d/kernel
<::2.bigger_auto_encoder/bigger_encoder/conv2d/bias
L:J 22bigger_auto_encoder/bigger_encoder/conv2d_1/kernel
>:< 20bigger_auto_encoder/bigger_encoder/conv2d_1/bias
L:J @22bigger_auto_encoder/bigger_encoder/conv2d_2/kernel
>:<@20bigger_auto_encoder/bigger_encoder/conv2d_2/bias
L:J@@22bigger_auto_encoder/bigger_encoder/conv2d_3/kernel
>:<@20bigger_auto_encoder/bigger_encoder/conv2d_3/bias
T:R@@2:bigger_auto_encoder/bigger_decoder/conv2d_transpose/kernel
F:D@28bigger_auto_encoder/bigger_decoder/conv2d_transpose/bias
V:T@@2<bigger_auto_encoder/bigger_decoder/conv2d_transpose_1/kernel
H:F@2:bigger_auto_encoder/bigger_decoder/conv2d_transpose_1/bias
V:T @2<bigger_auto_encoder/bigger_decoder/conv2d_transpose_2/kernel
H:F 2:bigger_auto_encoder/bigger_decoder/conv2d_transpose_2/bias
V:T 2<bigger_auto_encoder/bigger_decoder/conv2d_transpose_3/kernel
H:F2:bigger_auto_encoder/bigger_decoder/conv2d_transpose_3/bias
L:J22bigger_auto_encoder/bigger_decoder/conv2d_4/kernel
>:<20bigger_auto_encoder/bigger_decoder/conv2d_4/bias
.
0
1"
trackable_list_wrapper
 "
trackable_dict_wrapper
.
i0
j1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
 0
!1"
trackable_list_wrapper
.
 0
!1"
trackable_list_wrapper
�
7regularization_losses
knon_trainable_variables

llayers
mmetrics
8trainable_variables
nlayer_regularization_losses
olayer_metrics
9	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
"0
#1"
trackable_list_wrapper
.
"0
#1"
trackable_list_wrapper
�
;regularization_losses
pnon_trainable_variables

qlayers
rmetrics
<trainable_variables
slayer_regularization_losses
tlayer_metrics
=	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
$0
%1"
trackable_list_wrapper
.
$0
%1"
trackable_list_wrapper
�
?regularization_losses
unon_trainable_variables

vlayers
wmetrics
@trainable_variables
xlayer_regularization_losses
ylayer_metrics
A	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
&0
'1"
trackable_list_wrapper
.
&0
'1"
trackable_list_wrapper
�
Cregularization_losses
znon_trainable_variables

{layers
|metrics
Dtrainable_variables
}layer_regularization_losses
~layer_metrics
E	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
<
	0

1
2
3"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
(0
)1"
trackable_list_wrapper
.
(0
)1"
trackable_list_wrapper
�
Lregularization_losses
non_trainable_variables
�layers
�metrics
Mtrainable_variables
 �layer_regularization_losses
�layer_metrics
N	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
*0
+1"
trackable_list_wrapper
.
*0
+1"
trackable_list_wrapper
�
Pregularization_losses
�non_trainable_variables
�layers
�metrics
Qtrainable_variables
 �layer_regularization_losses
�layer_metrics
R	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
,0
-1"
trackable_list_wrapper
.
,0
-1"
trackable_list_wrapper
�
Tregularization_losses
�non_trainable_variables
�layers
�metrics
Utrainable_variables
 �layer_regularization_losses
�layer_metrics
V	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
.0
/1"
trackable_list_wrapper
.
.0
/1"
trackable_list_wrapper
�
Xregularization_losses
�non_trainable_variables
�layers
�metrics
Ytrainable_variables
 �layer_regularization_losses
�layer_metrics
Z	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
\regularization_losses
�non_trainable_variables
�layers
�metrics
]trainable_variables
 �layer_regularization_losses
�layer_metrics
^	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
00
11"
trackable_list_wrapper
.
00
11"
trackable_list_wrapper
�
`regularization_losses
�non_trainable_variables
�layers
�metrics
atrainable_variables
 �layer_regularization_losses
�layer_metrics
b	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
J
0
1
2
3
4
5"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�

�total

�count
�	variables
�	keras_api"�
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
�

�total

�count
�
_fn_kwargs
�	variables
�	keras_api"�
_tf_keras_metric�{"class_name": "MeanMetricWrapper", "name": "mae", "dtype": "float32", "config": {"name": "mae", "dtype": "float32", "fn": "mean_absolute_error"}}
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
:  (2total
:  (2count
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
O:M27Adam/bigger_auto_encoder/bigger_encoder/conv2d/kernel/m
A:?25Adam/bigger_auto_encoder/bigger_encoder/conv2d/bias/m
Q:O 29Adam/bigger_auto_encoder/bigger_encoder/conv2d_1/kernel/m
C:A 27Adam/bigger_auto_encoder/bigger_encoder/conv2d_1/bias/m
Q:O @29Adam/bigger_auto_encoder/bigger_encoder/conv2d_2/kernel/m
C:A@27Adam/bigger_auto_encoder/bigger_encoder/conv2d_2/bias/m
Q:O@@29Adam/bigger_auto_encoder/bigger_encoder/conv2d_3/kernel/m
C:A@27Adam/bigger_auto_encoder/bigger_encoder/conv2d_3/bias/m
Y:W@@2AAdam/bigger_auto_encoder/bigger_decoder/conv2d_transpose/kernel/m
K:I@2?Adam/bigger_auto_encoder/bigger_decoder/conv2d_transpose/bias/m
[:Y@@2CAdam/bigger_auto_encoder/bigger_decoder/conv2d_transpose_1/kernel/m
M:K@2AAdam/bigger_auto_encoder/bigger_decoder/conv2d_transpose_1/bias/m
[:Y @2CAdam/bigger_auto_encoder/bigger_decoder/conv2d_transpose_2/kernel/m
M:K 2AAdam/bigger_auto_encoder/bigger_decoder/conv2d_transpose_2/bias/m
[:Y 2CAdam/bigger_auto_encoder/bigger_decoder/conv2d_transpose_3/kernel/m
M:K2AAdam/bigger_auto_encoder/bigger_decoder/conv2d_transpose_3/bias/m
Q:O29Adam/bigger_auto_encoder/bigger_decoder/conv2d_4/kernel/m
C:A27Adam/bigger_auto_encoder/bigger_decoder/conv2d_4/bias/m
O:M27Adam/bigger_auto_encoder/bigger_encoder/conv2d/kernel/v
A:?25Adam/bigger_auto_encoder/bigger_encoder/conv2d/bias/v
Q:O 29Adam/bigger_auto_encoder/bigger_encoder/conv2d_1/kernel/v
C:A 27Adam/bigger_auto_encoder/bigger_encoder/conv2d_1/bias/v
Q:O @29Adam/bigger_auto_encoder/bigger_encoder/conv2d_2/kernel/v
C:A@27Adam/bigger_auto_encoder/bigger_encoder/conv2d_2/bias/v
Q:O@@29Adam/bigger_auto_encoder/bigger_encoder/conv2d_3/kernel/v
C:A@27Adam/bigger_auto_encoder/bigger_encoder/conv2d_3/bias/v
Y:W@@2AAdam/bigger_auto_encoder/bigger_decoder/conv2d_transpose/kernel/v
K:I@2?Adam/bigger_auto_encoder/bigger_decoder/conv2d_transpose/bias/v
[:Y@@2CAdam/bigger_auto_encoder/bigger_decoder/conv2d_transpose_1/kernel/v
M:K@2AAdam/bigger_auto_encoder/bigger_decoder/conv2d_transpose_1/bias/v
[:Y @2CAdam/bigger_auto_encoder/bigger_decoder/conv2d_transpose_2/kernel/v
M:K 2AAdam/bigger_auto_encoder/bigger_decoder/conv2d_transpose_2/bias/v
[:Y 2CAdam/bigger_auto_encoder/bigger_decoder/conv2d_transpose_3/kernel/v
M:K2AAdam/bigger_auto_encoder/bigger_decoder/conv2d_transpose_3/bias/v
Q:O29Adam/bigger_auto_encoder/bigger_decoder/conv2d_4/kernel/v
C:A27Adam/bigger_auto_encoder/bigger_decoder/conv2d_4/bias/v
�2�
!__inference__wrapped_model_143726�
���
FullArgSpec
args� 
varargsjargs
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *0�-
+�(
input_1�����������
�2�
4__inference_bigger_auto_encoder_layer_call_fn_144224�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *0�-
+�(
input_1�����������
�2�
O__inference_bigger_auto_encoder_layer_call_and_return_conditional_losses_144182�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *0�-
+�(
input_1�����������
�2�
/__inference_bigger_encoder_layer_call_fn_143861�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *0�-
+�(
input_1�����������
�2�
J__inference_bigger_encoder_layer_call_and_return_conditional_losses_143839�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *0�-
+�(
input_1�����������
�2�
/__inference_bigger_decoder_layer_call_fn_144139�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *0�-
+�(
input_1�����������@
�2�
J__inference_bigger_decoder_layer_call_and_return_conditional_losses_144113�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *0�-
+�(
input_1�����������@
�B�
$__inference_signature_wrapper_144275input_1"�
���
FullArgSpec
args� 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
'__inference_conv2d_layer_call_fn_144295�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
B__inference_conv2d_layer_call_and_return_conditional_losses_144286�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
)__inference_conv2d_1_layer_call_fn_144315�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
D__inference_conv2d_1_layer_call_and_return_conditional_losses_144306�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
)__inference_conv2d_2_layer_call_fn_144335�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
D__inference_conv2d_2_layer_call_and_return_conditional_losses_144326�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
)__inference_conv2d_3_layer_call_fn_144355�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
D__inference_conv2d_3_layer_call_and_return_conditional_losses_144346�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
1__inference_conv2d_transpose_layer_call_fn_143906�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *7�4
2�/+���������������������������@
�2�
L__inference_conv2d_transpose_layer_call_and_return_conditional_losses_143896�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *7�4
2�/+���������������������������@
�2�
3__inference_conv2d_transpose_1_layer_call_fn_143951�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *7�4
2�/+���������������������������@
�2�
N__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_143941�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *7�4
2�/+���������������������������@
�2�
3__inference_conv2d_transpose_2_layer_call_fn_143996�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *7�4
2�/+���������������������������@
�2�
N__inference_conv2d_transpose_2_layer_call_and_return_conditional_losses_143986�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *7�4
2�/+���������������������������@
�2�
3__inference_conv2d_transpose_3_layer_call_fn_144041�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *7�4
2�/+��������������������������� 
�2�
N__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_144031�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *7�4
2�/+��������������������������� 
�2�
.__inference_up_sampling2d_layer_call_fn_144060�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *@�=
;�84������������������������������������
�2�
I__inference_up_sampling2d_layer_call_and_return_conditional_losses_144054�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *@�=
;�84������������������������������������
�2�
)__inference_conv2d_4_layer_call_fn_144375�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
D__inference_conv2d_4_layer_call_and_return_conditional_losses_144366�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 �
!__inference__wrapped_model_143726� !"#$%&'()*+,-./01:�7
0�-
+�(
input_1�����������
� "=�:
8
output_1,�)
output_1������������
O__inference_bigger_auto_encoder_layer_call_and_return_conditional_losses_144182� !"#$%&'()*+,-./01:�7
0�-
+�(
input_1�����������
� "?�<
5�2
0+���������������������������
� �
4__inference_bigger_auto_encoder_layer_call_fn_144224� !"#$%&'()*+,-./01:�7
0�-
+�(
input_1�����������
� "2�/+����������������������������
J__inference_bigger_decoder_layer_call_and_return_conditional_losses_144113�
()*+,-./01:�7
0�-
+�(
input_1�����������@
� "?�<
5�2
0+���������������������������
� �
/__inference_bigger_decoder_layer_call_fn_144139|
()*+,-./01:�7
0�-
+�(
input_1�����������@
� "2�/+����������������������������
J__inference_bigger_encoder_layer_call_and_return_conditional_losses_143839w !"#$%&':�7
0�-
+�(
input_1�����������
� "/�,
%�"
0�����������@
� �
/__inference_bigger_encoder_layer_call_fn_143861j !"#$%&':�7
0�-
+�(
input_1�����������
� ""������������@�
D__inference_conv2d_1_layer_call_and_return_conditional_losses_144306p"#9�6
/�,
*�'
inputs�����������
� "/�,
%�"
0����������� 
� �
)__inference_conv2d_1_layer_call_fn_144315c"#9�6
/�,
*�'
inputs�����������
� ""������������ �
D__inference_conv2d_2_layer_call_and_return_conditional_losses_144326p$%9�6
/�,
*�'
inputs����������� 
� "/�,
%�"
0�����������@
� �
)__inference_conv2d_2_layer_call_fn_144335c$%9�6
/�,
*�'
inputs����������� 
� ""������������@�
D__inference_conv2d_3_layer_call_and_return_conditional_losses_144346p&'9�6
/�,
*�'
inputs�����������@
� "/�,
%�"
0�����������@
� �
)__inference_conv2d_3_layer_call_fn_144355c&'9�6
/�,
*�'
inputs�����������@
� ""������������@�
D__inference_conv2d_4_layer_call_and_return_conditional_losses_144366�01I�F
?�<
:�7
inputs+���������������������������
� "?�<
5�2
0+���������������������������
� �
)__inference_conv2d_4_layer_call_fn_144375�01I�F
?�<
:�7
inputs+���������������������������
� "2�/+����������������������������
B__inference_conv2d_layer_call_and_return_conditional_losses_144286p !9�6
/�,
*�'
inputs�����������
� "/�,
%�"
0�����������
� �
'__inference_conv2d_layer_call_fn_144295c !9�6
/�,
*�'
inputs�����������
� ""�������������
N__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_143941�*+I�F
?�<
:�7
inputs+���������������������������@
� "?�<
5�2
0+���������������������������@
� �
3__inference_conv2d_transpose_1_layer_call_fn_143951�*+I�F
?�<
:�7
inputs+���������������������������@
� "2�/+���������������������������@�
N__inference_conv2d_transpose_2_layer_call_and_return_conditional_losses_143986�,-I�F
?�<
:�7
inputs+���������������������������@
� "?�<
5�2
0+��������������������������� 
� �
3__inference_conv2d_transpose_2_layer_call_fn_143996�,-I�F
?�<
:�7
inputs+���������������������������@
� "2�/+��������������������������� �
N__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_144031�./I�F
?�<
:�7
inputs+��������������������������� 
� "?�<
5�2
0+���������������������������
� �
3__inference_conv2d_transpose_3_layer_call_fn_144041�./I�F
?�<
:�7
inputs+��������������������������� 
� "2�/+����������������������������
L__inference_conv2d_transpose_layer_call_and_return_conditional_losses_143896�()I�F
?�<
:�7
inputs+���������������������������@
� "?�<
5�2
0+���������������������������@
� �
1__inference_conv2d_transpose_layer_call_fn_143906�()I�F
?�<
:�7
inputs+���������������������������@
� "2�/+���������������������������@�
$__inference_signature_wrapper_144275� !"#$%&'()*+,-./01E�B
� 
;�8
6
input_1+�(
input_1�����������"=�:
8
output_1,�)
output_1������������
I__inference_up_sampling2d_layer_call_and_return_conditional_losses_144054�R�O
H�E
C�@
inputs4������������������������������������
� "H�E
>�;
04������������������������������������
� �
.__inference_up_sampling2d_layer_call_fn_144060�R�O
H�E
C�@
inputs4������������������������������������
� ";�84������������������������������������