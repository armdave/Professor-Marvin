ñ%
È
D
AddV2
x"T
y"T
z"T"
Ttype:
2	
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( 
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

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
û
FusedBatchNormV3
x"T

scale"U
offset"U	
mean"U
variance"U
y"T

batch_mean"U
batch_variance"U
reserve_space_1"U
reserve_space_2"U
reserve_space_3"U"
Ttype:
2"
Utype:
2"
epsilonfloat%·Ñ8"&
exponential_avg_factorfloat%  ?";
data_formatstringNHWC:
NHWCNCHWNDHWCNCDHW"
is_trainingbool(
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	

MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 ":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(
?
Mul
x"T
y"T
z"T"
Ttype:
2	
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
dtypetype
E
Relu
features"T
activations"T"
Ttype:
2	
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
.
Rsqrt
x"T
y"T"
Ttype:

2
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
9
Softmax
logits"T
softmax"T"
Ttype:
2
Á
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
executor_typestring ¨
@
StaticRegexFullMatch	
input

output
"
patternstring
÷
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

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
<
Sub
x"T
y"T
z"T"
Ttype:
2	

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.9.22v2.9.1-132-g18960c44ad38ø
p
dense_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_4/bias
i
 dense_4/bias/Read/ReadVariableOpReadVariableOpdense_4/bias*
_output_shapes
:*
dtype0
y
dense_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*
shared_namedense_4/kernel
r
"dense_4/kernel/Read/ReadVariableOpReadVariableOpdense_4/kernel*
_output_shapes
:	*
dtype0
¥
&batch_normalization_39/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&batch_normalization_39/moving_variance

:batch_normalization_39/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_39/moving_variance*
_output_shapes	
:*
dtype0

"batch_normalization_39/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"batch_normalization_39/moving_mean

6batch_normalization_39/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_39/moving_mean*
_output_shapes	
:*
dtype0

batch_normalization_39/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_39/beta

/batch_normalization_39/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_39/beta*
_output_shapes	
:*
dtype0

batch_normalization_39/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namebatch_normalization_39/gamma

0batch_normalization_39/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_39/gamma*
_output_shapes	
:*
dtype0
y
features256/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_namefeatures256/bias
r
$features256/bias/Read/ReadVariableOpReadVariableOpfeatures256/bias*
_output_shapes	
:*
dtype0

features256/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*#
shared_namefeatures256/kernel
{
&features256/kernel/Read/ReadVariableOpReadVariableOpfeatures256/kernel* 
_output_shapes
:
*
dtype0
¥
&batch_normalization_38/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&batch_normalization_38/moving_variance

:batch_normalization_38/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_38/moving_variance*
_output_shapes	
:*
dtype0

"batch_normalization_38/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"batch_normalization_38/moving_mean

6batch_normalization_38/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_38/moving_mean*
_output_shapes	
:*
dtype0

batch_normalization_38/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_38/beta

/batch_normalization_38/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_38/beta*
_output_shapes	
:*
dtype0

batch_normalization_38/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namebatch_normalization_38/gamma

0batch_normalization_38/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_38/gamma*
_output_shapes	
:*
dtype0
y
features512/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_namefeatures512/bias
r
$features512/bias/Read/ReadVariableOpReadVariableOpfeatures512/bias*
_output_shapes	
:*
dtype0

features512/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*#
shared_namefeatures512/kernel
{
&features512/kernel/Read/ReadVariableOpReadVariableOpfeatures512/kernel* 
_output_shapes
:
*
dtype0
¥
&batch_normalization_37/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&batch_normalization_37/moving_variance

:batch_normalization_37/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_37/moving_variance*
_output_shapes	
:*
dtype0

"batch_normalization_37/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"batch_normalization_37/moving_mean

6batch_normalization_37/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_37/moving_mean*
_output_shapes	
:*
dtype0

batch_normalization_37/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_37/beta

/batch_normalization_37/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_37/beta*
_output_shapes	
:*
dtype0

batch_normalization_37/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namebatch_normalization_37/gamma

0batch_normalization_37/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_37/gamma*
_output_shapes	
:*
dtype0
u
conv2d_24/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_24/bias
n
"conv2d_24/bias/Read/ReadVariableOpReadVariableOpconv2d_24/bias*
_output_shapes	
:*
dtype0

conv2d_24/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*!
shared_nameconv2d_24/kernel
~
$conv2d_24/kernel/Read/ReadVariableOpReadVariableOpconv2d_24/kernel*'
_output_shapes
:@*
dtype0
¤
&batch_normalization_36/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*7
shared_name(&batch_normalization_36/moving_variance

:batch_normalization_36/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_36/moving_variance*
_output_shapes
:@*
dtype0

"batch_normalization_36/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"batch_normalization_36/moving_mean

6batch_normalization_36/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_36/moving_mean*
_output_shapes
:@*
dtype0

batch_normalization_36/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_namebatch_normalization_36/beta

/batch_normalization_36/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_36/beta*
_output_shapes
:@*
dtype0

batch_normalization_36/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*-
shared_namebatch_normalization_36/gamma

0batch_normalization_36/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_36/gamma*
_output_shapes
:@*
dtype0
t
conv2d_23/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_23/bias
m
"conv2d_23/bias/Read/ReadVariableOpReadVariableOpconv2d_23/bias*
_output_shapes
:@*
dtype0

conv2d_23/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*!
shared_nameconv2d_23/kernel
}
$conv2d_23/kernel/Read/ReadVariableOpReadVariableOpconv2d_23/kernel*&
_output_shapes
: @*
dtype0
¤
&batch_normalization_35/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *7
shared_name(&batch_normalization_35/moving_variance

:batch_normalization_35/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_35/moving_variance*
_output_shapes
: *
dtype0

"batch_normalization_35/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"batch_normalization_35/moving_mean

6batch_normalization_35/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_35/moving_mean*
_output_shapes
: *
dtype0

batch_normalization_35/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *,
shared_namebatch_normalization_35/beta

/batch_normalization_35/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_35/beta*
_output_shapes
: *
dtype0

batch_normalization_35/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *-
shared_namebatch_normalization_35/gamma

0batch_normalization_35/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_35/gamma*
_output_shapes
: *
dtype0
t
conv2d_22/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_22/bias
m
"conv2d_22/bias/Read/ReadVariableOpReadVariableOpconv2d_22/bias*
_output_shapes
: *
dtype0

conv2d_22/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *!
shared_nameconv2d_22/kernel
}
$conv2d_22/kernel/Read/ReadVariableOpReadVariableOpconv2d_22/kernel*&
_output_shapes
: *
dtype0
¤
&batch_normalization_34/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&batch_normalization_34/moving_variance

:batch_normalization_34/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_34/moving_variance*
_output_shapes
:*
dtype0

"batch_normalization_34/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"batch_normalization_34/moving_mean

6batch_normalization_34/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_34/moving_mean*
_output_shapes
:*
dtype0

batch_normalization_34/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_34/beta

/batch_normalization_34/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_34/beta*
_output_shapes
:*
dtype0

batch_normalization_34/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namebatch_normalization_34/gamma

0batch_normalization_34/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_34/gamma*
_output_shapes
:*
dtype0
t
conv2d_21/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_21/bias
m
"conv2d_21/bias/Read/ReadVariableOpReadVariableOpconv2d_21/bias*
_output_shapes
:*
dtype0

conv2d_21/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv2d_21/kernel
}
$conv2d_21/kernel/Read/ReadVariableOpReadVariableOpconv2d_21/kernel*&
_output_shapes
:*
dtype0
¤
&batch_normalization_33/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&batch_normalization_33/moving_variance

:batch_normalization_33/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_33/moving_variance*
_output_shapes
:*
dtype0

"batch_normalization_33/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"batch_normalization_33/moving_mean

6batch_normalization_33/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_33/moving_mean*
_output_shapes
:*
dtype0

batch_normalization_33/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_33/beta

/batch_normalization_33/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_33/beta*
_output_shapes
:*
dtype0

batch_normalization_33/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namebatch_normalization_33/gamma

0batch_normalization_33/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_33/gamma*
_output_shapes
:*
dtype0
t
conv2d_20/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_20/bias
m
"conv2d_20/bias/Read/ReadVariableOpReadVariableOpconv2d_20/bias*
_output_shapes
:*
dtype0

conv2d_20/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv2d_20/kernel
}
$conv2d_20/kernel/Read/ReadVariableOpReadVariableOpconv2d_20/kernel*&
_output_shapes
:*
dtype0
¤
&batch_normalization_32/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&batch_normalization_32/moving_variance

:batch_normalization_32/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_32/moving_variance*
_output_shapes
:*
dtype0

"batch_normalization_32/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"batch_normalization_32/moving_mean

6batch_normalization_32/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_32/moving_mean*
_output_shapes
:*
dtype0

batch_normalization_32/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_32/beta

/batch_normalization_32/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_32/beta*
_output_shapes
:*
dtype0

batch_normalization_32/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namebatch_normalization_32/gamma

0batch_normalization_32/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_32/gamma*
_output_shapes
:*
dtype0

NoOpNoOp
Åç
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*ÿæ
valueôæBðæ Bèæ
ß
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer-4
layer-5
layer-6
layer_with_weights-3
layer-7
	layer_with_weights-4
	layer-8

layer-9
layer-10
layer-11
layer_with_weights-5
layer-12
layer_with_weights-6
layer-13
layer-14
layer-15
layer-16
layer_with_weights-7
layer-17
layer_with_weights-8
layer-18
layer-19
layer-20
layer-21
layer_with_weights-9
layer-22
layer_with_weights-10
layer-23
layer-24
layer-25
layer-26
layer-27
layer_with_weights-11
layer-28
layer_with_weights-12
layer-29
layer-30
 layer-31
!layer_with_weights-13
!layer-32
"layer_with_weights-14
"layer-33
#layer-34
$layer-35
%layer_with_weights-15
%layer-36
&	variables
'trainable_variables
(regularization_losses
)	keras_api
*__call__
*+&call_and_return_all_conditional_losses
,_default_save_signature
-
signatures*

.	variables
/trainable_variables
0regularization_losses
1	keras_api
2__call__
*3&call_and_return_all_conditional_losses* 
Õ
4	variables
5trainable_variables
6regularization_losses
7	keras_api
8__call__
*9&call_and_return_all_conditional_losses
:axis
	;gamma
<beta
=moving_mean
>moving_variance*
È
?	variables
@trainable_variables
Aregularization_losses
B	keras_api
C__call__
*D&call_and_return_all_conditional_losses

Ekernel
Fbias
 G_jit_compiled_convolution_op*
Õ
H	variables
Itrainable_variables
Jregularization_losses
K	keras_api
L__call__
*M&call_and_return_all_conditional_losses
Naxis
	Ogamma
Pbeta
Qmoving_mean
Rmoving_variance*

S	variables
Ttrainable_variables
Uregularization_losses
V	keras_api
W__call__
*X&call_and_return_all_conditional_losses* 

Y	variables
Ztrainable_variables
[regularization_losses
\	keras_api
]__call__
*^&call_and_return_all_conditional_losses* 
¥
_	variables
`trainable_variables
aregularization_losses
b	keras_api
c__call__
*d&call_and_return_all_conditional_losses
e_random_generator* 
È
f	variables
gtrainable_variables
hregularization_losses
i	keras_api
j__call__
*k&call_and_return_all_conditional_losses

lkernel
mbias
 n_jit_compiled_convolution_op*
Õ
o	variables
ptrainable_variables
qregularization_losses
r	keras_api
s__call__
*t&call_and_return_all_conditional_losses
uaxis
	vgamma
wbeta
xmoving_mean
ymoving_variance*

z	variables
{trainable_variables
|regularization_losses
}	keras_api
~__call__
*&call_and_return_all_conditional_losses* 

	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses* 
¬
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses
_random_generator* 
Ñ
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses
kernel
	bias
!_jit_compiled_convolution_op*
à
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses
	axis

gamma
	beta
moving_mean
 moving_variance*

¡	variables
¢trainable_variables
£regularization_losses
¤	keras_api
¥__call__
+¦&call_and_return_all_conditional_losses* 

§	variables
¨trainable_variables
©regularization_losses
ª	keras_api
«__call__
+¬&call_and_return_all_conditional_losses* 
¬
­	variables
®trainable_variables
¯regularization_losses
°	keras_api
±__call__
+²&call_and_return_all_conditional_losses
³_random_generator* 
Ñ
´	variables
µtrainable_variables
¶regularization_losses
·	keras_api
¸__call__
+¹&call_and_return_all_conditional_losses
ºkernel
	»bias
!¼_jit_compiled_convolution_op*
à
½	variables
¾trainable_variables
¿regularization_losses
À	keras_api
Á__call__
+Â&call_and_return_all_conditional_losses
	Ãaxis

Ägamma
	Åbeta
Æmoving_mean
Çmoving_variance*

È	variables
Étrainable_variables
Êregularization_losses
Ë	keras_api
Ì__call__
+Í&call_and_return_all_conditional_losses* 

Î	variables
Ïtrainable_variables
Ðregularization_losses
Ñ	keras_api
Ò__call__
+Ó&call_and_return_all_conditional_losses* 
¬
Ô	variables
Õtrainable_variables
Öregularization_losses
×	keras_api
Ø__call__
+Ù&call_and_return_all_conditional_losses
Ú_random_generator* 
Ñ
Û	variables
Ütrainable_variables
Ýregularization_losses
Þ	keras_api
ß__call__
+à&call_and_return_all_conditional_losses
ákernel
	âbias
!ã_jit_compiled_convolution_op*
à
ä	variables
åtrainable_variables
æregularization_losses
ç	keras_api
è__call__
+é&call_and_return_all_conditional_losses
	êaxis

ëgamma
	ìbeta
ímoving_mean
îmoving_variance*

ï	variables
ðtrainable_variables
ñregularization_losses
ò	keras_api
ó__call__
+ô&call_and_return_all_conditional_losses* 

õ	variables
ötrainable_variables
÷regularization_losses
ø	keras_api
ù__call__
+ú&call_and_return_all_conditional_losses* 
¬
û	variables
ütrainable_variables
ýregularization_losses
þ	keras_api
ÿ__call__
+&call_and_return_all_conditional_losses
_random_generator* 

	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses* 
®
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses
kernel
	bias*
à
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses
	axis

gamma
	beta
moving_mean
moving_variance*

	variables
trainable_variables
regularization_losses
	keras_api
__call__
+ &call_and_return_all_conditional_losses* 
¬
¡	variables
¢trainable_variables
£regularization_losses
¤	keras_api
¥__call__
+¦&call_and_return_all_conditional_losses
§_random_generator* 
®
¨	variables
©trainable_variables
ªregularization_losses
«	keras_api
¬__call__
+­&call_and_return_all_conditional_losses
®kernel
	¯bias*
à
°	variables
±trainable_variables
²regularization_losses
³	keras_api
´__call__
+µ&call_and_return_all_conditional_losses
	¶axis

·gamma
	¸beta
¹moving_mean
ºmoving_variance*

»	variables
¼trainable_variables
½regularization_losses
¾	keras_api
¿__call__
+À&call_and_return_all_conditional_losses* 
¬
Á	variables
Âtrainable_variables
Ãregularization_losses
Ä	keras_api
Å__call__
+Æ&call_and_return_all_conditional_losses
Ç_random_generator* 
®
È	variables
Étrainable_variables
Êregularization_losses
Ë	keras_api
Ì__call__
+Í&call_and_return_all_conditional_losses
Îkernel
	Ïbias*

;0
<1
=2
>3
E4
F5
O6
P7
Q8
R9
l10
m11
v12
w13
x14
y15
16
17
18
19
20
 21
º22
»23
Ä24
Å25
Æ26
Ç27
á28
â29
ë30
ì31
í32
î33
34
35
36
37
38
39
®40
¯41
·42
¸43
¹44
º45
Î46
Ï47*

;0
<1
E2
F3
O4
P5
l6
m7
v8
w9
10
11
12
13
º14
»15
Ä16
Å17
á18
â19
ë20
ì21
22
23
24
25
®26
¯27
·28
¸29
Î30
Ï31*
* 
µ
Ðnon_trainable_variables
Ñlayers
Òmetrics
 Ólayer_regularization_losses
Ôlayer_metrics
&	variables
'trainable_variables
(regularization_losses
*__call__
,_default_save_signature
*+&call_and_return_all_conditional_losses
&+"call_and_return_conditional_losses*
:
Õtrace_0
Ötrace_1
×trace_2
Øtrace_3* 
:
Ùtrace_0
Útrace_1
Ûtrace_2
Ütrace_3* 
* 

Ýserving_default* 
* 
* 
* 

Þnon_trainable_variables
ßlayers
àmetrics
 álayer_regularization_losses
âlayer_metrics
.	variables
/trainable_variables
0regularization_losses
2__call__
*3&call_and_return_all_conditional_losses
&3"call_and_return_conditional_losses* 

ãtrace_0* 

ätrace_0* 
 
;0
<1
=2
>3*

;0
<1*
* 

ånon_trainable_variables
ælayers
çmetrics
 èlayer_regularization_losses
élayer_metrics
4	variables
5trainable_variables
6regularization_losses
8__call__
*9&call_and_return_all_conditional_losses
&9"call_and_return_conditional_losses*

êtrace_0
ëtrace_1* 

ìtrace_0
ítrace_1* 
* 
ke
VARIABLE_VALUEbatch_normalization_32/gamma5layer_with_weights-0/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEbatch_normalization_32/beta4layer_with_weights-0/beta/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUE"batch_normalization_32/moving_mean;layer_with_weights-0/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE&batch_normalization_32/moving_variance?layer_with_weights-0/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*

E0
F1*

E0
F1*
* 

înon_trainable_variables
ïlayers
ðmetrics
 ñlayer_regularization_losses
òlayer_metrics
?	variables
@trainable_variables
Aregularization_losses
C__call__
*D&call_and_return_all_conditional_losses
&D"call_and_return_conditional_losses*

ótrace_0* 

ôtrace_0* 
`Z
VARIABLE_VALUEconv2d_20/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEconv2d_20/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
 
O0
P1
Q2
R3*

O0
P1*
* 

õnon_trainable_variables
ölayers
÷metrics
 ølayer_regularization_losses
ùlayer_metrics
H	variables
Itrainable_variables
Jregularization_losses
L__call__
*M&call_and_return_all_conditional_losses
&M"call_and_return_conditional_losses*

útrace_0
ûtrace_1* 

ütrace_0
ýtrace_1* 
* 
ke
VARIABLE_VALUEbatch_normalization_33/gamma5layer_with_weights-2/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEbatch_normalization_33/beta4layer_with_weights-2/beta/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUE"batch_normalization_33/moving_mean;layer_with_weights-2/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE&batch_normalization_33/moving_variance?layer_with_weights-2/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 

þnon_trainable_variables
ÿlayers
metrics
 layer_regularization_losses
layer_metrics
S	variables
Ttrainable_variables
Uregularization_losses
W__call__
*X&call_and_return_all_conditional_losses
&X"call_and_return_conditional_losses* 

trace_0* 

trace_0* 
* 
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
Y	variables
Ztrainable_variables
[regularization_losses
]__call__
*^&call_and_return_all_conditional_losses
&^"call_and_return_conditional_losses* 

trace_0* 

trace_0* 
* 
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
_	variables
`trainable_variables
aregularization_losses
c__call__
*d&call_and_return_all_conditional_losses
&d"call_and_return_conditional_losses* 

trace_0
trace_1* 

trace_0
trace_1* 
* 

l0
m1*

l0
m1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
f	variables
gtrainable_variables
hregularization_losses
j__call__
*k&call_and_return_all_conditional_losses
&k"call_and_return_conditional_losses*

trace_0* 

trace_0* 
`Z
VARIABLE_VALUEconv2d_21/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEconv2d_21/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
 
v0
w1
x2
y3*

v0
w1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
 layer_metrics
o	variables
ptrainable_variables
qregularization_losses
s__call__
*t&call_and_return_all_conditional_losses
&t"call_and_return_conditional_losses*

¡trace_0
¢trace_1* 

£trace_0
¤trace_1* 
* 
ke
VARIABLE_VALUEbatch_normalization_34/gamma5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEbatch_normalization_34/beta4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUE"batch_normalization_34/moving_mean;layer_with_weights-4/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE&batch_normalization_34/moving_variance?layer_with_weights-4/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 

¥non_trainable_variables
¦layers
§metrics
 ¨layer_regularization_losses
©layer_metrics
z	variables
{trainable_variables
|regularization_losses
~__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses* 

ªtrace_0* 

«trace_0* 
* 
* 
* 

¬non_trainable_variables
­layers
®metrics
 ¯layer_regularization_losses
°layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses* 

±trace_0* 

²trace_0* 
* 
* 
* 

³non_trainable_variables
´layers
µmetrics
 ¶layer_regularization_losses
·layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses* 

¸trace_0
¹trace_1* 

ºtrace_0
»trace_1* 
* 

0
1*

0
1*
* 

¼non_trainable_variables
½layers
¾metrics
 ¿layer_regularization_losses
Àlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*

Átrace_0* 

Âtrace_0* 
`Z
VARIABLE_VALUEconv2d_22/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEconv2d_22/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
$
0
1
2
 3*

0
1*
* 

Ãnon_trainable_variables
Älayers
Åmetrics
 Ælayer_regularization_losses
Çlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*

Ètrace_0
Étrace_1* 

Êtrace_0
Ëtrace_1* 
* 
ke
VARIABLE_VALUEbatch_normalization_35/gamma5layer_with_weights-6/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEbatch_normalization_35/beta4layer_with_weights-6/beta/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUE"batch_normalization_35/moving_mean;layer_with_weights-6/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE&batch_normalization_35/moving_variance?layer_with_weights-6/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 

Ìnon_trainable_variables
Ílayers
Îmetrics
 Ïlayer_regularization_losses
Ðlayer_metrics
¡	variables
¢trainable_variables
£regularization_losses
¥__call__
+¦&call_and_return_all_conditional_losses
'¦"call_and_return_conditional_losses* 

Ñtrace_0* 

Òtrace_0* 
* 
* 
* 

Ónon_trainable_variables
Ôlayers
Õmetrics
 Ölayer_regularization_losses
×layer_metrics
§	variables
¨trainable_variables
©regularization_losses
«__call__
+¬&call_and_return_all_conditional_losses
'¬"call_and_return_conditional_losses* 

Øtrace_0* 

Ùtrace_0* 
* 
* 
* 

Únon_trainable_variables
Ûlayers
Ümetrics
 Ýlayer_regularization_losses
Þlayer_metrics
­	variables
®trainable_variables
¯regularization_losses
±__call__
+²&call_and_return_all_conditional_losses
'²"call_and_return_conditional_losses* 

ßtrace_0
àtrace_1* 

átrace_0
âtrace_1* 
* 

º0
»1*

º0
»1*
* 

ãnon_trainable_variables
älayers
åmetrics
 ælayer_regularization_losses
çlayer_metrics
´	variables
µtrainable_variables
¶regularization_losses
¸__call__
+¹&call_and_return_all_conditional_losses
'¹"call_and_return_conditional_losses*

ètrace_0* 

étrace_0* 
`Z
VARIABLE_VALUEconv2d_23/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEconv2d_23/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
$
Ä0
Å1
Æ2
Ç3*

Ä0
Å1*
* 

ênon_trainable_variables
ëlayers
ìmetrics
 ílayer_regularization_losses
îlayer_metrics
½	variables
¾trainable_variables
¿regularization_losses
Á__call__
+Â&call_and_return_all_conditional_losses
'Â"call_and_return_conditional_losses*

ïtrace_0
ðtrace_1* 

ñtrace_0
òtrace_1* 
* 
ke
VARIABLE_VALUEbatch_normalization_36/gamma5layer_with_weights-8/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEbatch_normalization_36/beta4layer_with_weights-8/beta/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUE"batch_normalization_36/moving_mean;layer_with_weights-8/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE&batch_normalization_36/moving_variance?layer_with_weights-8/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 

ónon_trainable_variables
ôlayers
õmetrics
 ölayer_regularization_losses
÷layer_metrics
È	variables
Étrainable_variables
Êregularization_losses
Ì__call__
+Í&call_and_return_all_conditional_losses
'Í"call_and_return_conditional_losses* 

øtrace_0* 

ùtrace_0* 
* 
* 
* 

únon_trainable_variables
ûlayers
ümetrics
 ýlayer_regularization_losses
þlayer_metrics
Î	variables
Ïtrainable_variables
Ðregularization_losses
Ò__call__
+Ó&call_and_return_all_conditional_losses
'Ó"call_and_return_conditional_losses* 

ÿtrace_0* 

trace_0* 
* 
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
Ô	variables
Õtrainable_variables
Öregularization_losses
Ø__call__
+Ù&call_and_return_all_conditional_losses
'Ù"call_and_return_conditional_losses* 

trace_0
trace_1* 

trace_0
trace_1* 
* 

á0
â1*

á0
â1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
Û	variables
Ütrainable_variables
Ýregularization_losses
ß__call__
+à&call_and_return_all_conditional_losses
'à"call_and_return_conditional_losses*

trace_0* 

trace_0* 
`Z
VARIABLE_VALUEconv2d_24/kernel6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEconv2d_24/bias4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
$
ë0
ì1
í2
î3*

ë0
ì1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
ä	variables
åtrainable_variables
æregularization_losses
è__call__
+é&call_and_return_all_conditional_losses
'é"call_and_return_conditional_losses*

trace_0
trace_1* 

trace_0
trace_1* 
* 
lf
VARIABLE_VALUEbatch_normalization_37/gamma6layer_with_weights-10/gamma/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEbatch_normalization_37/beta5layer_with_weights-10/beta/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUE"batch_normalization_37/moving_mean<layer_with_weights-10/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUE&batch_normalization_37/moving_variance@layer_with_weights-10/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
ï	variables
ðtrainable_variables
ñregularization_losses
ó__call__
+ô&call_and_return_all_conditional_losses
'ô"call_and_return_conditional_losses* 

trace_0* 

 trace_0* 
* 
* 
* 

¡non_trainable_variables
¢layers
£metrics
 ¤layer_regularization_losses
¥layer_metrics
õ	variables
ötrainable_variables
÷regularization_losses
ù__call__
+ú&call_and_return_all_conditional_losses
'ú"call_and_return_conditional_losses* 

¦trace_0* 

§trace_0* 
* 
* 
* 

¨non_trainable_variables
©layers
ªmetrics
 «layer_regularization_losses
¬layer_metrics
û	variables
ütrainable_variables
ýregularization_losses
ÿ__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses* 

­trace_0
®trace_1* 

¯trace_0
°trace_1* 
* 
* 
* 
* 

±non_trainable_variables
²layers
³metrics
 ´layer_regularization_losses
µlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses* 

¶trace_0* 

·trace_0* 

0
1*

0
1*
* 

¸non_trainable_variables
¹layers
ºmetrics
 »layer_regularization_losses
¼layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*

½trace_0* 

¾trace_0* 
c]
VARIABLE_VALUEfeatures512/kernel7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEfeatures512/bias5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUE*
$
0
1
2
3*

0
1*
* 

¿non_trainable_variables
Àlayers
Ámetrics
 Âlayer_regularization_losses
Ãlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*

Ätrace_0
Åtrace_1* 

Ætrace_0
Çtrace_1* 
* 
lf
VARIABLE_VALUEbatch_normalization_38/gamma6layer_with_weights-12/gamma/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEbatch_normalization_38/beta5layer_with_weights-12/beta/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUE"batch_normalization_38/moving_mean<layer_with_weights-12/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUE&batch_normalization_38/moving_variance@layer_with_weights-12/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 

Ènon_trainable_variables
Élayers
Êmetrics
 Ëlayer_regularization_losses
Ìlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+ &call_and_return_all_conditional_losses
' "call_and_return_conditional_losses* 

Ítrace_0* 

Îtrace_0* 
* 
* 
* 

Ïnon_trainable_variables
Ðlayers
Ñmetrics
 Òlayer_regularization_losses
Ólayer_metrics
¡	variables
¢trainable_variables
£regularization_losses
¥__call__
+¦&call_and_return_all_conditional_losses
'¦"call_and_return_conditional_losses* 

Ôtrace_0
Õtrace_1* 

Ötrace_0
×trace_1* 
* 

®0
¯1*

®0
¯1*
* 

Ønon_trainable_variables
Ùlayers
Úmetrics
 Ûlayer_regularization_losses
Ülayer_metrics
¨	variables
©trainable_variables
ªregularization_losses
¬__call__
+­&call_and_return_all_conditional_losses
'­"call_and_return_conditional_losses*

Ýtrace_0* 

Þtrace_0* 
c]
VARIABLE_VALUEfeatures256/kernel7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEfeatures256/bias5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUE*
$
·0
¸1
¹2
º3*

·0
¸1*
* 

ßnon_trainable_variables
àlayers
ámetrics
 âlayer_regularization_losses
ãlayer_metrics
°	variables
±trainable_variables
²regularization_losses
´__call__
+µ&call_and_return_all_conditional_losses
'µ"call_and_return_conditional_losses*

ätrace_0
åtrace_1* 

ætrace_0
çtrace_1* 
* 
lf
VARIABLE_VALUEbatch_normalization_39/gamma6layer_with_weights-14/gamma/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEbatch_normalization_39/beta5layer_with_weights-14/beta/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUE"batch_normalization_39/moving_mean<layer_with_weights-14/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUE&batch_normalization_39/moving_variance@layer_with_weights-14/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 

ènon_trainable_variables
élayers
êmetrics
 ëlayer_regularization_losses
ìlayer_metrics
»	variables
¼trainable_variables
½regularization_losses
¿__call__
+À&call_and_return_all_conditional_losses
'À"call_and_return_conditional_losses* 

ítrace_0* 

îtrace_0* 
* 
* 
* 

ïnon_trainable_variables
ðlayers
ñmetrics
 òlayer_regularization_losses
ólayer_metrics
Á	variables
Âtrainable_variables
Ãregularization_losses
Å__call__
+Æ&call_and_return_all_conditional_losses
'Æ"call_and_return_conditional_losses* 

ôtrace_0
õtrace_1* 

ötrace_0
÷trace_1* 
* 

Î0
Ï1*

Î0
Ï1*
* 

ønon_trainable_variables
ùlayers
úmetrics
 ûlayer_regularization_losses
ülayer_metrics
È	variables
Étrainable_variables
Êregularization_losses
Ì__call__
+Í&call_and_return_all_conditional_losses
'Í"call_and_return_conditional_losses*

ýtrace_0* 

þtrace_0* 
_Y
VARIABLE_VALUEdense_4/kernel7layer_with_weights-15/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_4/bias5layer_with_weights-15/bias/.ATTRIBUTES/VARIABLE_VALUE*

=0
>1
Q2
R3
x4
y5
6
 7
Æ8
Ç9
í10
î11
12
13
¹14
º15*
¢
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
 31
!32
"33
#34
$35
%36*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

=0
>1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

Q0
R1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

x0
y1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

0
 1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

Æ0
Ç1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

í0
î1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

0
1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

¹0
º1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

serving_default_reshape_4_inputPlaceholder*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿc(*
dtype0* 
shape:ÿÿÿÿÿÿÿÿÿc(
è
StatefulPartitionedCallStatefulPartitionedCallserving_default_reshape_4_inputbatch_normalization_32/gammabatch_normalization_32/beta"batch_normalization_32/moving_mean&batch_normalization_32/moving_varianceconv2d_20/kernelconv2d_20/biasbatch_normalization_33/gammabatch_normalization_33/beta"batch_normalization_33/moving_mean&batch_normalization_33/moving_varianceconv2d_21/kernelconv2d_21/biasbatch_normalization_34/gammabatch_normalization_34/beta"batch_normalization_34/moving_mean&batch_normalization_34/moving_varianceconv2d_22/kernelconv2d_22/biasbatch_normalization_35/gammabatch_normalization_35/beta"batch_normalization_35/moving_mean&batch_normalization_35/moving_varianceconv2d_23/kernelconv2d_23/biasbatch_normalization_36/gammabatch_normalization_36/beta"batch_normalization_36/moving_mean&batch_normalization_36/moving_varianceconv2d_24/kernelconv2d_24/biasbatch_normalization_37/gammabatch_normalization_37/beta"batch_normalization_37/moving_mean&batch_normalization_37/moving_variancefeatures512/kernelfeatures512/bias&batch_normalization_38/moving_variancebatch_normalization_38/gamma"batch_normalization_38/moving_meanbatch_normalization_38/betafeatures256/kernelfeatures256/bias&batch_normalization_39/moving_variancebatch_normalization_39/gamma"batch_normalization_39/moving_meanbatch_normalization_39/betadense_4/kerneldense_4/bias*<
Tin5
321*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*R
_read_only_resource_inputs4
20	
 !"#$%&'()*+,-./0*0
config_proto 

CPU

GPU2*0J 8 *,
f'R%
#__inference_signature_wrapper_17340
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
Ù
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename0batch_normalization_32/gamma/Read/ReadVariableOp/batch_normalization_32/beta/Read/ReadVariableOp6batch_normalization_32/moving_mean/Read/ReadVariableOp:batch_normalization_32/moving_variance/Read/ReadVariableOp$conv2d_20/kernel/Read/ReadVariableOp"conv2d_20/bias/Read/ReadVariableOp0batch_normalization_33/gamma/Read/ReadVariableOp/batch_normalization_33/beta/Read/ReadVariableOp6batch_normalization_33/moving_mean/Read/ReadVariableOp:batch_normalization_33/moving_variance/Read/ReadVariableOp$conv2d_21/kernel/Read/ReadVariableOp"conv2d_21/bias/Read/ReadVariableOp0batch_normalization_34/gamma/Read/ReadVariableOp/batch_normalization_34/beta/Read/ReadVariableOp6batch_normalization_34/moving_mean/Read/ReadVariableOp:batch_normalization_34/moving_variance/Read/ReadVariableOp$conv2d_22/kernel/Read/ReadVariableOp"conv2d_22/bias/Read/ReadVariableOp0batch_normalization_35/gamma/Read/ReadVariableOp/batch_normalization_35/beta/Read/ReadVariableOp6batch_normalization_35/moving_mean/Read/ReadVariableOp:batch_normalization_35/moving_variance/Read/ReadVariableOp$conv2d_23/kernel/Read/ReadVariableOp"conv2d_23/bias/Read/ReadVariableOp0batch_normalization_36/gamma/Read/ReadVariableOp/batch_normalization_36/beta/Read/ReadVariableOp6batch_normalization_36/moving_mean/Read/ReadVariableOp:batch_normalization_36/moving_variance/Read/ReadVariableOp$conv2d_24/kernel/Read/ReadVariableOp"conv2d_24/bias/Read/ReadVariableOp0batch_normalization_37/gamma/Read/ReadVariableOp/batch_normalization_37/beta/Read/ReadVariableOp6batch_normalization_37/moving_mean/Read/ReadVariableOp:batch_normalization_37/moving_variance/Read/ReadVariableOp&features512/kernel/Read/ReadVariableOp$features512/bias/Read/ReadVariableOp0batch_normalization_38/gamma/Read/ReadVariableOp/batch_normalization_38/beta/Read/ReadVariableOp6batch_normalization_38/moving_mean/Read/ReadVariableOp:batch_normalization_38/moving_variance/Read/ReadVariableOp&features256/kernel/Read/ReadVariableOp$features256/bias/Read/ReadVariableOp0batch_normalization_39/gamma/Read/ReadVariableOp/batch_normalization_39/beta/Read/ReadVariableOp6batch_normalization_39/moving_mean/Read/ReadVariableOp:batch_normalization_39/moving_variance/Read/ReadVariableOp"dense_4/kernel/Read/ReadVariableOp dense_4/bias/Read/ReadVariableOpConst*=
Tin6
422*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *'
f"R 
__inference__traced_save_19210

StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamebatch_normalization_32/gammabatch_normalization_32/beta"batch_normalization_32/moving_mean&batch_normalization_32/moving_varianceconv2d_20/kernelconv2d_20/biasbatch_normalization_33/gammabatch_normalization_33/beta"batch_normalization_33/moving_mean&batch_normalization_33/moving_varianceconv2d_21/kernelconv2d_21/biasbatch_normalization_34/gammabatch_normalization_34/beta"batch_normalization_34/moving_mean&batch_normalization_34/moving_varianceconv2d_22/kernelconv2d_22/biasbatch_normalization_35/gammabatch_normalization_35/beta"batch_normalization_35/moving_mean&batch_normalization_35/moving_varianceconv2d_23/kernelconv2d_23/biasbatch_normalization_36/gammabatch_normalization_36/beta"batch_normalization_36/moving_mean&batch_normalization_36/moving_varianceconv2d_24/kernelconv2d_24/biasbatch_normalization_37/gammabatch_normalization_37/beta"batch_normalization_37/moving_mean&batch_normalization_37/moving_variancefeatures512/kernelfeatures512/biasbatch_normalization_38/gammabatch_normalization_38/beta"batch_normalization_38/moving_mean&batch_normalization_38/moving_variancefeatures256/kernelfeatures256/biasbatch_normalization_39/gammabatch_normalization_39/beta"batch_normalization_39/moving_mean&batch_normalization_39/moving_variancedense_4/kerneldense_4/bias*<
Tin5
321*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 **
f%R#
!__inference__traced_restore_19364Úô
ø
c
E__inference_dropout_30_layer_call_and_return_conditional_losses_15943

inputs

identity_1V
IdentityIdentityinputs*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ c

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ :W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
»

d
E__inference_dropout_32_layer_call_and_return_conditional_losses_18740

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *«ªª?m
dropout/MulMulinputsdropout/Const:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿC
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  >¯
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
IdentityIdentitydropout/Mul_1:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

À
Q__inference_batch_normalization_35_layer_call_and_return_conditional_losses_15454

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity¢AssignNewValue¢AssignNewValue_1¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0Ö
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : : :*
epsilon%o:*
exponential_avg_factor%
×#<Æ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(Ð
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ Ô
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
±
Õ
6__inference_batch_normalization_39_layer_call_fn_18932

inputs
unknown:	
	unknown_0:	
	unknown_1:	
	unknown_2:	
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_batch_normalization_39_layer_call_and_return_conditional_losses_15782p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ñý
Ã+
G__inference_sequential_4_layer_call_and_return_conditional_losses_17742

inputs<
.batch_normalization_32_readvariableop_resource:>
0batch_normalization_32_readvariableop_1_resource:M
?batch_normalization_32_fusedbatchnormv3_readvariableop_resource:O
Abatch_normalization_32_fusedbatchnormv3_readvariableop_1_resource:B
(conv2d_20_conv2d_readvariableop_resource:7
)conv2d_20_biasadd_readvariableop_resource:<
.batch_normalization_33_readvariableop_resource:>
0batch_normalization_33_readvariableop_1_resource:M
?batch_normalization_33_fusedbatchnormv3_readvariableop_resource:O
Abatch_normalization_33_fusedbatchnormv3_readvariableop_1_resource:B
(conv2d_21_conv2d_readvariableop_resource:7
)conv2d_21_biasadd_readvariableop_resource:<
.batch_normalization_34_readvariableop_resource:>
0batch_normalization_34_readvariableop_1_resource:M
?batch_normalization_34_fusedbatchnormv3_readvariableop_resource:O
Abatch_normalization_34_fusedbatchnormv3_readvariableop_1_resource:B
(conv2d_22_conv2d_readvariableop_resource: 7
)conv2d_22_biasadd_readvariableop_resource: <
.batch_normalization_35_readvariableop_resource: >
0batch_normalization_35_readvariableop_1_resource: M
?batch_normalization_35_fusedbatchnormv3_readvariableop_resource: O
Abatch_normalization_35_fusedbatchnormv3_readvariableop_1_resource: B
(conv2d_23_conv2d_readvariableop_resource: @7
)conv2d_23_biasadd_readvariableop_resource:@<
.batch_normalization_36_readvariableop_resource:@>
0batch_normalization_36_readvariableop_1_resource:@M
?batch_normalization_36_fusedbatchnormv3_readvariableop_resource:@O
Abatch_normalization_36_fusedbatchnormv3_readvariableop_1_resource:@C
(conv2d_24_conv2d_readvariableop_resource:@8
)conv2d_24_biasadd_readvariableop_resource:	=
.batch_normalization_37_readvariableop_resource:	?
0batch_normalization_37_readvariableop_1_resource:	N
?batch_normalization_37_fusedbatchnormv3_readvariableop_resource:	P
Abatch_normalization_37_fusedbatchnormv3_readvariableop_1_resource:	>
*features512_matmul_readvariableop_resource:
:
+features512_biasadd_readvariableop_resource:	G
8batch_normalization_38_batchnorm_readvariableop_resource:	K
<batch_normalization_38_batchnorm_mul_readvariableop_resource:	I
:batch_normalization_38_batchnorm_readvariableop_1_resource:	I
:batch_normalization_38_batchnorm_readvariableop_2_resource:	>
*features256_matmul_readvariableop_resource:
:
+features256_biasadd_readvariableop_resource:	G
8batch_normalization_39_batchnorm_readvariableop_resource:	K
<batch_normalization_39_batchnorm_mul_readvariableop_resource:	I
:batch_normalization_39_batchnorm_readvariableop_1_resource:	I
:batch_normalization_39_batchnorm_readvariableop_2_resource:	9
&dense_4_matmul_readvariableop_resource:	5
'dense_4_biasadd_readvariableop_resource:
identity¢6batch_normalization_32/FusedBatchNormV3/ReadVariableOp¢8batch_normalization_32/FusedBatchNormV3/ReadVariableOp_1¢%batch_normalization_32/ReadVariableOp¢'batch_normalization_32/ReadVariableOp_1¢6batch_normalization_33/FusedBatchNormV3/ReadVariableOp¢8batch_normalization_33/FusedBatchNormV3/ReadVariableOp_1¢%batch_normalization_33/ReadVariableOp¢'batch_normalization_33/ReadVariableOp_1¢6batch_normalization_34/FusedBatchNormV3/ReadVariableOp¢8batch_normalization_34/FusedBatchNormV3/ReadVariableOp_1¢%batch_normalization_34/ReadVariableOp¢'batch_normalization_34/ReadVariableOp_1¢6batch_normalization_35/FusedBatchNormV3/ReadVariableOp¢8batch_normalization_35/FusedBatchNormV3/ReadVariableOp_1¢%batch_normalization_35/ReadVariableOp¢'batch_normalization_35/ReadVariableOp_1¢6batch_normalization_36/FusedBatchNormV3/ReadVariableOp¢8batch_normalization_36/FusedBatchNormV3/ReadVariableOp_1¢%batch_normalization_36/ReadVariableOp¢'batch_normalization_36/ReadVariableOp_1¢6batch_normalization_37/FusedBatchNormV3/ReadVariableOp¢8batch_normalization_37/FusedBatchNormV3/ReadVariableOp_1¢%batch_normalization_37/ReadVariableOp¢'batch_normalization_37/ReadVariableOp_1¢/batch_normalization_38/batchnorm/ReadVariableOp¢1batch_normalization_38/batchnorm/ReadVariableOp_1¢1batch_normalization_38/batchnorm/ReadVariableOp_2¢3batch_normalization_38/batchnorm/mul/ReadVariableOp¢/batch_normalization_39/batchnorm/ReadVariableOp¢1batch_normalization_39/batchnorm/ReadVariableOp_1¢1batch_normalization_39/batchnorm/ReadVariableOp_2¢3batch_normalization_39/batchnorm/mul/ReadVariableOp¢ conv2d_20/BiasAdd/ReadVariableOp¢conv2d_20/Conv2D/ReadVariableOp¢ conv2d_21/BiasAdd/ReadVariableOp¢conv2d_21/Conv2D/ReadVariableOp¢ conv2d_22/BiasAdd/ReadVariableOp¢conv2d_22/Conv2D/ReadVariableOp¢ conv2d_23/BiasAdd/ReadVariableOp¢conv2d_23/Conv2D/ReadVariableOp¢ conv2d_24/BiasAdd/ReadVariableOp¢conv2d_24/Conv2D/ReadVariableOp¢dense_4/BiasAdd/ReadVariableOp¢dense_4/MatMul/ReadVariableOp¢"features256/BiasAdd/ReadVariableOp¢!features256/MatMul/ReadVariableOp¢"features512/BiasAdd/ReadVariableOp¢!features512/MatMul/ReadVariableOpE
reshape_4/ShapeShapeinputs*
T0*
_output_shapes
:g
reshape_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: i
reshape_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
reshape_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
reshape_4/strided_sliceStridedSlicereshape_4/Shape:output:0&reshape_4/strided_slice/stack:output:0(reshape_4/strided_slice/stack_1:output:0(reshape_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask[
reshape_4/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :c[
reshape_4/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :([
reshape_4/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :Û
reshape_4/Reshape/shapePack reshape_4/strided_slice:output:0"reshape_4/Reshape/shape/1:output:0"reshape_4/Reshape/shape/2:output:0"reshape_4/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:
reshape_4/ReshapeReshapeinputs reshape_4/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿc(
%batch_normalization_32/ReadVariableOpReadVariableOp.batch_normalization_32_readvariableop_resource*
_output_shapes
:*
dtype0
'batch_normalization_32/ReadVariableOp_1ReadVariableOp0batch_normalization_32_readvariableop_1_resource*
_output_shapes
:*
dtype0²
6batch_normalization_32/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_32_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0¶
8batch_normalization_32/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_32_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0½
'batch_normalization_32/FusedBatchNormV3FusedBatchNormV3reshape_4/Reshape:output:0-batch_normalization_32/ReadVariableOp:value:0/batch_normalization_32/ReadVariableOp_1:value:0>batch_normalization_32/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_32/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿc(:::::*
epsilon%o:*
is_training( 
conv2d_20/Conv2D/ReadVariableOpReadVariableOp(conv2d_20_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0Ò
conv2d_20/Conv2DConv2D+batch_normalization_32/FusedBatchNormV3:y:0'conv2d_20/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿc(*
paddingSAME*
strides

 conv2d_20/BiasAdd/ReadVariableOpReadVariableOp)conv2d_20_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
conv2d_20/BiasAddBiasAddconv2d_20/Conv2D:output:0(conv2d_20/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿc(
%batch_normalization_33/ReadVariableOpReadVariableOp.batch_normalization_33_readvariableop_resource*
_output_shapes
:*
dtype0
'batch_normalization_33/ReadVariableOp_1ReadVariableOp0batch_normalization_33_readvariableop_1_resource*
_output_shapes
:*
dtype0²
6batch_normalization_33/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_33_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0¶
8batch_normalization_33/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_33_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0½
'batch_normalization_33/FusedBatchNormV3FusedBatchNormV3conv2d_20/BiasAdd:output:0-batch_normalization_33/ReadVariableOp:value:0/batch_normalization_33/ReadVariableOp_1:value:0>batch_normalization_33/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_33/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿc(:::::*
epsilon%o:*
is_training( |
re_lu_28/ReluRelu+batch_normalization_33/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿc(­
max_pooling2d_20/MaxPoolMaxPoolre_lu_28/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*
ksize
*
paddingVALID*
strides
|
dropout_28/IdentityIdentity!max_pooling2d_20/MaxPool:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
conv2d_21/Conv2D/ReadVariableOpReadVariableOp(conv2d_21_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0Ã
conv2d_21/Conv2DConv2Ddropout_28/Identity:output:0'conv2d_21/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*
paddingSAME*
strides

 conv2d_21/BiasAdd/ReadVariableOpReadVariableOp)conv2d_21_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
conv2d_21/BiasAddBiasAddconv2d_21/Conv2D:output:0(conv2d_21/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
%batch_normalization_34/ReadVariableOpReadVariableOp.batch_normalization_34_readvariableop_resource*
_output_shapes
:*
dtype0
'batch_normalization_34/ReadVariableOp_1ReadVariableOp0batch_normalization_34_readvariableop_1_resource*
_output_shapes
:*
dtype0²
6batch_normalization_34/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_34_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0¶
8batch_normalization_34/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_34_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0½
'batch_normalization_34/FusedBatchNormV3FusedBatchNormV3conv2d_21/BiasAdd:output:0-batch_normalization_34/ReadVariableOp:value:0/batch_normalization_34/ReadVariableOp_1:value:0>batch_normalization_34/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_34/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ1:::::*
epsilon%o:*
is_training( |
re_lu_29/ReluRelu+batch_normalization_34/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1­
max_pooling2d_21/MaxPoolMaxPoolre_lu_29/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
ksize
*
paddingVALID*
strides
|
dropout_29/IdentityIdentity!max_pooling2d_21/MaxPool:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

conv2d_22/Conv2D/ReadVariableOpReadVariableOp(conv2d_22_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0Ã
conv2d_22/Conv2DConv2Ddropout_29/Identity:output:0'conv2d_22/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 *
paddingSAME*
strides

 conv2d_22/BiasAdd/ReadVariableOpReadVariableOp)conv2d_22_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0
conv2d_22/BiasAddBiasAddconv2d_22/Conv2D:output:0(conv2d_22/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
%batch_normalization_35/ReadVariableOpReadVariableOp.batch_normalization_35_readvariableop_resource*
_output_shapes
: *
dtype0
'batch_normalization_35/ReadVariableOp_1ReadVariableOp0batch_normalization_35_readvariableop_1_resource*
_output_shapes
: *
dtype0²
6batch_normalization_35/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_35_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0¶
8batch_normalization_35/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_35_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0½
'batch_normalization_35/FusedBatchNormV3FusedBatchNormV3conv2d_22/BiasAdd:output:0-batch_normalization_35/ReadVariableOp:value:0/batch_normalization_35/ReadVariableOp_1:value:0>batch_normalization_35/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_35/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ
 : : : : :*
epsilon%o:*
is_training( |
re_lu_30/ReluRelu+batch_normalization_35/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 ­
max_pooling2d_22/MaxPoolMaxPoolre_lu_30/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
ksize
*
paddingVALID*
strides
|
dropout_30/IdentityIdentity!max_pooling2d_22/MaxPool:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
conv2d_23/Conv2D/ReadVariableOpReadVariableOp(conv2d_23_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0Ã
conv2d_23/Conv2DConv2Ddropout_30/Identity:output:0'conv2d_23/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingSAME*
strides

 conv2d_23/BiasAdd/ReadVariableOpReadVariableOp)conv2d_23_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
conv2d_23/BiasAddBiasAddconv2d_23/Conv2D:output:0(conv2d_23/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
%batch_normalization_36/ReadVariableOpReadVariableOp.batch_normalization_36_readvariableop_resource*
_output_shapes
:@*
dtype0
'batch_normalization_36/ReadVariableOp_1ReadVariableOp0batch_normalization_36_readvariableop_1_resource*
_output_shapes
:@*
dtype0²
6batch_normalization_36/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_36_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0¶
8batch_normalization_36/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_36_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0½
'batch_normalization_36/FusedBatchNormV3FusedBatchNormV3conv2d_23/BiasAdd:output:0-batch_normalization_36/ReadVariableOp:value:0/batch_normalization_36/ReadVariableOp_1:value:0>batch_normalization_36/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_36/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
is_training( |
re_lu_31/ReluRelu+batch_normalization_36/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@­
max_pooling2d_23/MaxPoolMaxPoolre_lu_31/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
ksize
*
paddingVALID*
strides
|
dropout_31/IdentityIdentity!max_pooling2d_23/MaxPool:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
conv2d_24/Conv2D/ReadVariableOpReadVariableOp(conv2d_24_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype0Ä
conv2d_24/Conv2DConv2Ddropout_31/Identity:output:0'conv2d_24/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides

 conv2d_24/BiasAdd/ReadVariableOpReadVariableOp)conv2d_24_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
conv2d_24/BiasAddBiasAddconv2d_24/Conv2D:output:0(conv2d_24/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%batch_normalization_37/ReadVariableOpReadVariableOp.batch_normalization_37_readvariableop_resource*
_output_shapes	
:*
dtype0
'batch_normalization_37/ReadVariableOp_1ReadVariableOp0batch_normalization_37_readvariableop_1_resource*
_output_shapes	
:*
dtype0³
6batch_normalization_37/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_37_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0·
8batch_normalization_37/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_37_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0Â
'batch_normalization_37/FusedBatchNormV3FusedBatchNormV3conv2d_24/BiasAdd:output:0-batch_normalization_37/ReadVariableOp:value:0/batch_normalization_37/ReadVariableOp_1:value:0>batch_normalization_37/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_37/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( }
re_lu_32/ReluRelu+batch_normalization_37/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ®
max_pooling2d_24/MaxPoolMaxPoolre_lu_32/Relu:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
}
dropout_32/IdentityIdentity!max_pooling2d_24/MaxPool:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
flatten_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ  
flatten_4/ReshapeReshapedropout_32/Identity:output:0flatten_4/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!features512/MatMul/ReadVariableOpReadVariableOp*features512_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0
features512/MatMulMatMulflatten_4/Reshape:output:0)features512/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"features512/BiasAdd/ReadVariableOpReadVariableOp+features512_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
features512/BiasAddBiasAddfeatures512/MatMul:product:0*features512/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥
/batch_normalization_38/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_38_batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype0k
&batch_normalization_38/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:½
$batch_normalization_38/batchnorm/addAddV27batch_normalization_38/batchnorm/ReadVariableOp:value:0/batch_normalization_38/batchnorm/add/y:output:0*
T0*
_output_shapes	
:
&batch_normalization_38/batchnorm/RsqrtRsqrt(batch_normalization_38/batchnorm/add:z:0*
T0*
_output_shapes	
:­
3batch_normalization_38/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_38_batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype0º
$batch_normalization_38/batchnorm/mulMul*batch_normalization_38/batchnorm/Rsqrt:y:0;batch_normalization_38/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:¨
&batch_normalization_38/batchnorm/mul_1Mulfeatures512/BiasAdd:output:0(batch_normalization_38/batchnorm/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ©
1batch_normalization_38/batchnorm/ReadVariableOp_1ReadVariableOp:batch_normalization_38_batchnorm_readvariableop_1_resource*
_output_shapes	
:*
dtype0¸
&batch_normalization_38/batchnorm/mul_2Mul9batch_normalization_38/batchnorm/ReadVariableOp_1:value:0(batch_normalization_38/batchnorm/mul:z:0*
T0*
_output_shapes	
:©
1batch_normalization_38/batchnorm/ReadVariableOp_2ReadVariableOp:batch_normalization_38_batchnorm_readvariableop_2_resource*
_output_shapes	
:*
dtype0¸
$batch_normalization_38/batchnorm/subSub9batch_normalization_38/batchnorm/ReadVariableOp_2:value:0*batch_normalization_38/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:¸
&batch_normalization_38/batchnorm/add_1AddV2*batch_normalization_38/batchnorm/mul_1:z:0(batch_normalization_38/batchnorm/sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿt
re_lu_33/ReluRelu*batch_normalization_38/batchnorm/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿo
dropout_33/IdentityIdentityre_lu_33/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!features256/MatMul/ReadVariableOpReadVariableOp*features256_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0
features256/MatMulMatMuldropout_33/Identity:output:0)features256/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"features256/BiasAdd/ReadVariableOpReadVariableOp+features256_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
features256/BiasAddBiasAddfeatures256/MatMul:product:0*features256/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥
/batch_normalization_39/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_39_batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype0k
&batch_normalization_39/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:½
$batch_normalization_39/batchnorm/addAddV27batch_normalization_39/batchnorm/ReadVariableOp:value:0/batch_normalization_39/batchnorm/add/y:output:0*
T0*
_output_shapes	
:
&batch_normalization_39/batchnorm/RsqrtRsqrt(batch_normalization_39/batchnorm/add:z:0*
T0*
_output_shapes	
:­
3batch_normalization_39/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_39_batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype0º
$batch_normalization_39/batchnorm/mulMul*batch_normalization_39/batchnorm/Rsqrt:y:0;batch_normalization_39/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:¨
&batch_normalization_39/batchnorm/mul_1Mulfeatures256/BiasAdd:output:0(batch_normalization_39/batchnorm/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ©
1batch_normalization_39/batchnorm/ReadVariableOp_1ReadVariableOp:batch_normalization_39_batchnorm_readvariableop_1_resource*
_output_shapes	
:*
dtype0¸
&batch_normalization_39/batchnorm/mul_2Mul9batch_normalization_39/batchnorm/ReadVariableOp_1:value:0(batch_normalization_39/batchnorm/mul:z:0*
T0*
_output_shapes	
:©
1batch_normalization_39/batchnorm/ReadVariableOp_2ReadVariableOp:batch_normalization_39_batchnorm_readvariableop_2_resource*
_output_shapes	
:*
dtype0¸
$batch_normalization_39/batchnorm/subSub9batch_normalization_39/batchnorm/ReadVariableOp_2:value:0*batch_normalization_39/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:¸
&batch_normalization_39/batchnorm/add_1AddV2*batch_normalization_39/batchnorm/mul_1:z:0(batch_normalization_39/batchnorm/sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿt
re_lu_34/ReluRelu*batch_normalization_39/batchnorm/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿo
dropout_34/IdentityIdentityre_lu_34/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_4/MatMul/ReadVariableOpReadVariableOp&dense_4_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0
dense_4/MatMulMatMuldropout_34/Identity:output:0%dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_4/BiasAddBiasAdddense_4/MatMul:product:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
dense_4/SoftmaxSoftmaxdense_4/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
IdentityIdentitydense_4/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¶
NoOpNoOp7^batch_normalization_32/FusedBatchNormV3/ReadVariableOp9^batch_normalization_32/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_32/ReadVariableOp(^batch_normalization_32/ReadVariableOp_17^batch_normalization_33/FusedBatchNormV3/ReadVariableOp9^batch_normalization_33/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_33/ReadVariableOp(^batch_normalization_33/ReadVariableOp_17^batch_normalization_34/FusedBatchNormV3/ReadVariableOp9^batch_normalization_34/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_34/ReadVariableOp(^batch_normalization_34/ReadVariableOp_17^batch_normalization_35/FusedBatchNormV3/ReadVariableOp9^batch_normalization_35/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_35/ReadVariableOp(^batch_normalization_35/ReadVariableOp_17^batch_normalization_36/FusedBatchNormV3/ReadVariableOp9^batch_normalization_36/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_36/ReadVariableOp(^batch_normalization_36/ReadVariableOp_17^batch_normalization_37/FusedBatchNormV3/ReadVariableOp9^batch_normalization_37/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_37/ReadVariableOp(^batch_normalization_37/ReadVariableOp_10^batch_normalization_38/batchnorm/ReadVariableOp2^batch_normalization_38/batchnorm/ReadVariableOp_12^batch_normalization_38/batchnorm/ReadVariableOp_24^batch_normalization_38/batchnorm/mul/ReadVariableOp0^batch_normalization_39/batchnorm/ReadVariableOp2^batch_normalization_39/batchnorm/ReadVariableOp_12^batch_normalization_39/batchnorm/ReadVariableOp_24^batch_normalization_39/batchnorm/mul/ReadVariableOp!^conv2d_20/BiasAdd/ReadVariableOp ^conv2d_20/Conv2D/ReadVariableOp!^conv2d_21/BiasAdd/ReadVariableOp ^conv2d_21/Conv2D/ReadVariableOp!^conv2d_22/BiasAdd/ReadVariableOp ^conv2d_22/Conv2D/ReadVariableOp!^conv2d_23/BiasAdd/ReadVariableOp ^conv2d_23/Conv2D/ReadVariableOp!^conv2d_24/BiasAdd/ReadVariableOp ^conv2d_24/Conv2D/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp#^features256/BiasAdd/ReadVariableOp"^features256/MatMul/ReadVariableOp#^features512/BiasAdd/ReadVariableOp"^features512/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapesy
w:ÿÿÿÿÿÿÿÿÿc(: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2p
6batch_normalization_32/FusedBatchNormV3/ReadVariableOp6batch_normalization_32/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_32/FusedBatchNormV3/ReadVariableOp_18batch_normalization_32/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_32/ReadVariableOp%batch_normalization_32/ReadVariableOp2R
'batch_normalization_32/ReadVariableOp_1'batch_normalization_32/ReadVariableOp_12p
6batch_normalization_33/FusedBatchNormV3/ReadVariableOp6batch_normalization_33/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_33/FusedBatchNormV3/ReadVariableOp_18batch_normalization_33/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_33/ReadVariableOp%batch_normalization_33/ReadVariableOp2R
'batch_normalization_33/ReadVariableOp_1'batch_normalization_33/ReadVariableOp_12p
6batch_normalization_34/FusedBatchNormV3/ReadVariableOp6batch_normalization_34/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_34/FusedBatchNormV3/ReadVariableOp_18batch_normalization_34/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_34/ReadVariableOp%batch_normalization_34/ReadVariableOp2R
'batch_normalization_34/ReadVariableOp_1'batch_normalization_34/ReadVariableOp_12p
6batch_normalization_35/FusedBatchNormV3/ReadVariableOp6batch_normalization_35/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_35/FusedBatchNormV3/ReadVariableOp_18batch_normalization_35/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_35/ReadVariableOp%batch_normalization_35/ReadVariableOp2R
'batch_normalization_35/ReadVariableOp_1'batch_normalization_35/ReadVariableOp_12p
6batch_normalization_36/FusedBatchNormV3/ReadVariableOp6batch_normalization_36/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_36/FusedBatchNormV3/ReadVariableOp_18batch_normalization_36/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_36/ReadVariableOp%batch_normalization_36/ReadVariableOp2R
'batch_normalization_36/ReadVariableOp_1'batch_normalization_36/ReadVariableOp_12p
6batch_normalization_37/FusedBatchNormV3/ReadVariableOp6batch_normalization_37/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_37/FusedBatchNormV3/ReadVariableOp_18batch_normalization_37/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_37/ReadVariableOp%batch_normalization_37/ReadVariableOp2R
'batch_normalization_37/ReadVariableOp_1'batch_normalization_37/ReadVariableOp_12b
/batch_normalization_38/batchnorm/ReadVariableOp/batch_normalization_38/batchnorm/ReadVariableOp2f
1batch_normalization_38/batchnorm/ReadVariableOp_11batch_normalization_38/batchnorm/ReadVariableOp_12f
1batch_normalization_38/batchnorm/ReadVariableOp_21batch_normalization_38/batchnorm/ReadVariableOp_22j
3batch_normalization_38/batchnorm/mul/ReadVariableOp3batch_normalization_38/batchnorm/mul/ReadVariableOp2b
/batch_normalization_39/batchnorm/ReadVariableOp/batch_normalization_39/batchnorm/ReadVariableOp2f
1batch_normalization_39/batchnorm/ReadVariableOp_11batch_normalization_39/batchnorm/ReadVariableOp_12f
1batch_normalization_39/batchnorm/ReadVariableOp_21batch_normalization_39/batchnorm/ReadVariableOp_22j
3batch_normalization_39/batchnorm/mul/ReadVariableOp3batch_normalization_39/batchnorm/mul/ReadVariableOp2D
 conv2d_20/BiasAdd/ReadVariableOp conv2d_20/BiasAdd/ReadVariableOp2B
conv2d_20/Conv2D/ReadVariableOpconv2d_20/Conv2D/ReadVariableOp2D
 conv2d_21/BiasAdd/ReadVariableOp conv2d_21/BiasAdd/ReadVariableOp2B
conv2d_21/Conv2D/ReadVariableOpconv2d_21/Conv2D/ReadVariableOp2D
 conv2d_22/BiasAdd/ReadVariableOp conv2d_22/BiasAdd/ReadVariableOp2B
conv2d_22/Conv2D/ReadVariableOpconv2d_22/Conv2D/ReadVariableOp2D
 conv2d_23/BiasAdd/ReadVariableOp conv2d_23/BiasAdd/ReadVariableOp2B
conv2d_23/Conv2D/ReadVariableOpconv2d_23/Conv2D/ReadVariableOp2D
 conv2d_24/BiasAdd/ReadVariableOp conv2d_24/BiasAdd/ReadVariableOp2B
conv2d_24/Conv2D/ReadVariableOpconv2d_24/Conv2D/ReadVariableOp2@
dense_4/BiasAdd/ReadVariableOpdense_4/BiasAdd/ReadVariableOp2>
dense_4/MatMul/ReadVariableOpdense_4/MatMul/ReadVariableOp2H
"features256/BiasAdd/ReadVariableOp"features256/BiasAdd/ReadVariableOp2F
!features256/MatMul/ReadVariableOp!features256/MatMul/ReadVariableOp2H
"features512/BiasAdd/ReadVariableOp"features512/BiasAdd/ReadVariableOp2F
!features512/MatMul/ReadVariableOp!features512/MatMul/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿc(
 
_user_specified_nameinputs

À
Q__inference_batch_normalization_35_layer_call_and_return_conditional_losses_18437

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity¢AssignNewValue¢AssignNewValue_1¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0Ö
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : : :*
epsilon%o:*
exponential_avg_factor%
×#<Æ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(Ð
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ Ô
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
Ó
`
D__inference_reshape_4_layer_call_and_return_conditional_losses_18038

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :cQ
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :(Q
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :©
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿc(`
IdentityIdentityReshape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿc("
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿc(:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿc(
 
_user_specified_nameinputs
³

d
E__inference_dropout_30_layer_call_and_return_conditional_losses_18484

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *«ªª?l
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  >®
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ w
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ q
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ a
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ :W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
ç
_
C__inference_re_lu_30_layer_call_and_return_conditional_losses_18447

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 b
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ
 :W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
 
_user_specified_nameinputs
ç
_
C__inference_re_lu_30_layer_call_and_return_conditional_losses_15935

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 b
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ
 :W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
 
_user_specified_nameinputs
È
`
D__inference_flatten_4_layer_call_and_return_conditional_losses_18751

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ  ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
³

d
E__inference_dropout_28_layer_call_and_return_conditional_losses_18228

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *«ªª?l
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  >®
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1w
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1q
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1a
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ1:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
 
_user_specified_nameinputs
	
Ñ
6__inference_batch_normalization_33_layer_call_fn_18132

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_batch_normalization_33_layer_call_and_return_conditional_losses_15271
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


,__inference_sequential_4_layer_call_fn_17542

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:#
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:#
	unknown_9:

unknown_10:

unknown_11:

unknown_12:

unknown_13:

unknown_14:$

unknown_15: 

unknown_16: 

unknown_17: 

unknown_18: 

unknown_19: 

unknown_20: $

unknown_21: @

unknown_22:@

unknown_23:@

unknown_24:@

unknown_25:@

unknown_26:@%

unknown_27:@

unknown_28:	

unknown_29:	

unknown_30:	

unknown_31:	

unknown_32:	

unknown_33:


unknown_34:	

unknown_35:	

unknown_36:	

unknown_37:	

unknown_38:	

unknown_39:


unknown_40:	

unknown_41:	

unknown_42:	

unknown_43:	

unknown_44:	

unknown_45:	

unknown_46:
identity¢StatefulPartitionedCallÊ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46*<
Tin5
321*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*B
_read_only_resource_inputs$
"  #$'()*-./0*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_sequential_4_layer_call_and_return_conditional_losses_16763o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapesy
w:ÿÿÿÿÿÿÿÿÿc(: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿc(
 
_user_specified_nameinputs
¯%
î
Q__inference_batch_normalization_39_layer_call_and_return_conditional_losses_15782

inputs6
'assignmovingavg_readvariableop_resource:	8
)assignmovingavg_1_readvariableop_resource:	4
%batchnorm_mul_readvariableop_resource:	0
!batchnorm_readvariableop_resource:	
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	*
	keep_dims(e
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	*
	keep_dims(n
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:*
squeeze_dims
 t
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes	
:*
dtype0
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes	
:y
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:¬
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes	
:*
dtype0
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes	
:
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:´
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:r
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:Q
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype0u
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:d
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:w
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype0q
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:s
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿê
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ø
c
E__inference_dropout_28_layer_call_and_return_conditional_losses_18216

inputs

identity_1V
IdentityIdentityinputs*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1c

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ1:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
 
_user_specified_nameinputs
ç
_
C__inference_re_lu_31_layer_call_and_return_conditional_losses_18575

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@b
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
»

d
E__inference_dropout_32_layer_call_and_return_conditional_losses_16342

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *«ªª?m
dropout/MulMulinputsdropout/Const:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿC
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  >¯
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
IdentityIdentitydropout/Mul_1:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
²

,__inference_sequential_4_layer_call_fn_16963
reshape_4_input
unknown:
	unknown_0:
	unknown_1:
	unknown_2:#
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:#
	unknown_9:

unknown_10:

unknown_11:

unknown_12:

unknown_13:

unknown_14:$

unknown_15: 

unknown_16: 

unknown_17: 

unknown_18: 

unknown_19: 

unknown_20: $

unknown_21: @

unknown_22:@

unknown_23:@

unknown_24:@

unknown_25:@

unknown_26:@%

unknown_27:@

unknown_28:	

unknown_29:	

unknown_30:	

unknown_31:	

unknown_32:	

unknown_33:


unknown_34:	

unknown_35:	

unknown_36:	

unknown_37:	

unknown_38:	

unknown_39:


unknown_40:	

unknown_41:	

unknown_42:	

unknown_43:	

unknown_44:	

unknown_45:	

unknown_46:
identity¢StatefulPartitionedCallÓ
StatefulPartitionedCallStatefulPartitionedCallreshape_4_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46*<
Tin5
321*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*B
_read_only_resource_inputs$
"  #$'()*-./0*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_sequential_4_layer_call_and_return_conditional_losses_16763o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapesy
w:ÿÿÿÿÿÿÿÿÿc(: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿc(
)
_user_specified_namereshape_4_input
ø
c
E__inference_dropout_28_layer_call_and_return_conditional_losses_15863

inputs

identity_1V
IdentityIdentityinputs*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1c

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ1:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
 
_user_specified_nameinputs
í

)__inference_conv2d_20_layer_call_fn_18109

inputs!
unknown:
	unknown_0:
identity¢StatefulPartitionedCallä
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿc(*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_20_layer_call_and_return_conditional_losses_15835w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿc(`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿc(: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿc(
 
_user_specified_nameinputs
¾
D
(__inference_re_lu_28_layer_call_fn_18186

inputs
identity¹
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿc(* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_re_lu_28_layer_call_and_return_conditional_losses_15855h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿc("
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿc(:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿc(
 
_user_specified_nameinputs
§

ý
D__inference_conv2d_22_layer_call_and_return_conditional_losses_18375

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ
: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs

À
Q__inference_batch_normalization_32_layer_call_and_return_conditional_losses_18100

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity¢AssignNewValue¢AssignNewValue_1¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0Ö
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
exponential_avg_factor%
×#<Æ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(Ð
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÔ
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
	
Ñ
6__inference_batch_normalization_35_layer_call_fn_18388

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_batch_normalization_35_layer_call_and_return_conditional_losses_15423
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
	
Ñ
6__inference_batch_normalization_35_layer_call_fn_18401

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_batch_normalization_35_layer_call_and_return_conditional_losses_15454
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
Ü
c
E__inference_dropout_33_layer_call_and_return_conditional_losses_16070

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
í

)__inference_conv2d_22_layer_call_fn_18365

inputs!
unknown: 
	unknown_0: 
identity¢StatefulPartitionedCallä
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_22_layer_call_and_return_conditional_losses_15915w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ
: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs
ü
c
E__inference_dropout_32_layer_call_and_return_conditional_losses_16023

inputs

identity_1W
IdentityIdentityinputs*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿd

Identity_1IdentityIdentity:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

c
*__inference_dropout_32_layer_call_fn_18723

inputs
identity¢StatefulPartitionedCallÌ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_32_layer_call_and_return_conditional_losses_16342x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

À
Q__inference_batch_normalization_36_layer_call_and_return_conditional_losses_18565

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity¢AssignNewValue¢AssignNewValue_1¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0Ö
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
exponential_avg_factor%
×#<Æ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(Ð
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@Ô
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs

Ä
Q__inference_batch_normalization_37_layer_call_and_return_conditional_losses_15606

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identity¢AssignNewValue¢AssignNewValue_1¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0Û
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
exponential_avg_factor%
×#<Æ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(Ð
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÔ
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


#__inference_signature_wrapper_17340
reshape_4_input
unknown:
	unknown_0:
	unknown_1:
	unknown_2:#
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:#
	unknown_9:

unknown_10:

unknown_11:

unknown_12:

unknown_13:

unknown_14:$

unknown_15: 

unknown_16: 

unknown_17: 

unknown_18: 

unknown_19: 

unknown_20: $

unknown_21: @

unknown_22:@

unknown_23:@

unknown_24:@

unknown_25:@

unknown_26:@%

unknown_27:@

unknown_28:	

unknown_29:	

unknown_30:	

unknown_31:	

unknown_32:	

unknown_33:


unknown_34:	

unknown_35:	

unknown_36:	

unknown_37:	

unknown_38:	

unknown_39:


unknown_40:	

unknown_41:	

unknown_42:	

unknown_43:	

unknown_44:	

unknown_45:	

unknown_46:
identity¢StatefulPartitionedCall¼
StatefulPartitionedCallStatefulPartitionedCallreshape_4_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46*<
Tin5
321*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*R
_read_only_resource_inputs4
20	
 !"#$%&'()*+,-./0*0
config_proto 

CPU

GPU2*0J 8 *)
f$R"
 __inference__wrapped_model_15185o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapesy
w:ÿÿÿÿÿÿÿÿÿc(: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿc(
)
_user_specified_namereshape_4_input
ø
c
*__inference_dropout_33_layer_call_fn_18870

inputs
identity¢StatefulPartitionedCallÄ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_33_layer_call_and_return_conditional_losses_16297p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
³

d
E__inference_dropout_31_layer_call_and_return_conditional_losses_18612

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *«ªª?l
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  >®
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@w
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@q
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@a
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Ð

+__inference_features512_layer_call_fn_18760

inputs
unknown:

	unknown_0:	
identity¢StatefulPartitionedCallß
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_features512_layer_call_and_return_conditional_losses_16043p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

c
*__inference_dropout_30_layer_call_fn_18467

inputs
identity¢StatefulPartitionedCallË
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_30_layer_call_and_return_conditional_losses_16420w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs

g
K__inference_max_pooling2d_24_layer_call_and_return_conditional_losses_15626

inputs
identity¢
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ç
_
C__inference_re_lu_29_layer_call_and_return_conditional_losses_18319

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1b
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ1:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
 
_user_specified_nameinputs
Ð 
ê
G__inference_sequential_4_layer_call_and_return_conditional_losses_17100
reshape_4_input*
batch_normalization_32_16967:*
batch_normalization_32_16969:*
batch_normalization_32_16971:*
batch_normalization_32_16973:)
conv2d_20_16976:
conv2d_20_16978:*
batch_normalization_33_16981:*
batch_normalization_33_16983:*
batch_normalization_33_16985:*
batch_normalization_33_16987:)
conv2d_21_16993:
conv2d_21_16995:*
batch_normalization_34_16998:*
batch_normalization_34_17000:*
batch_normalization_34_17002:*
batch_normalization_34_17004:)
conv2d_22_17010: 
conv2d_22_17012: *
batch_normalization_35_17015: *
batch_normalization_35_17017: *
batch_normalization_35_17019: *
batch_normalization_35_17021: )
conv2d_23_17027: @
conv2d_23_17029:@*
batch_normalization_36_17032:@*
batch_normalization_36_17034:@*
batch_normalization_36_17036:@*
batch_normalization_36_17038:@*
conv2d_24_17044:@
conv2d_24_17046:	+
batch_normalization_37_17049:	+
batch_normalization_37_17051:	+
batch_normalization_37_17053:	+
batch_normalization_37_17055:	%
features512_17062:
 
features512_17064:	+
batch_normalization_38_17067:	+
batch_normalization_38_17069:	+
batch_normalization_38_17071:	+
batch_normalization_38_17073:	%
features256_17078:
 
features256_17080:	+
batch_normalization_39_17083:	+
batch_normalization_39_17085:	+
batch_normalization_39_17087:	+
batch_normalization_39_17089:	 
dense_4_17094:	
dense_4_17096:
identity¢.batch_normalization_32/StatefulPartitionedCall¢.batch_normalization_33/StatefulPartitionedCall¢.batch_normalization_34/StatefulPartitionedCall¢.batch_normalization_35/StatefulPartitionedCall¢.batch_normalization_36/StatefulPartitionedCall¢.batch_normalization_37/StatefulPartitionedCall¢.batch_normalization_38/StatefulPartitionedCall¢.batch_normalization_39/StatefulPartitionedCall¢!conv2d_20/StatefulPartitionedCall¢!conv2d_21/StatefulPartitionedCall¢!conv2d_22/StatefulPartitionedCall¢!conv2d_23/StatefulPartitionedCall¢!conv2d_24/StatefulPartitionedCall¢dense_4/StatefulPartitionedCall¢#features256/StatefulPartitionedCall¢#features512/StatefulPartitionedCallÍ
reshape_4/PartitionedCallPartitionedCallreshape_4_input*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿc(* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_reshape_4_layer_call_and_return_conditional_losses_15814
.batch_normalization_32/StatefulPartitionedCallStatefulPartitionedCall"reshape_4/PartitionedCall:output:0batch_normalization_32_16967batch_normalization_32_16969batch_normalization_32_16971batch_normalization_32_16973*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿc(*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_batch_normalization_32_layer_call_and_return_conditional_losses_15207­
!conv2d_20/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_32/StatefulPartitionedCall:output:0conv2d_20_16976conv2d_20_16978*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿc(*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_20_layer_call_and_return_conditional_losses_15835
.batch_normalization_33/StatefulPartitionedCallStatefulPartitionedCall*conv2d_20/StatefulPartitionedCall:output:0batch_normalization_33_16981batch_normalization_33_16983batch_normalization_33_16985batch_normalization_33_16987*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿc(*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_batch_normalization_33_layer_call_and_return_conditional_losses_15271ó
re_lu_28/PartitionedCallPartitionedCall7batch_normalization_33/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿc(* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_re_lu_28_layer_call_and_return_conditional_losses_15855í
 max_pooling2d_20/PartitionedCallPartitionedCall!re_lu_28/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_max_pooling2d_20_layer_call_and_return_conditional_losses_15322é
dropout_28/PartitionedCallPartitionedCall)max_pooling2d_20/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_28_layer_call_and_return_conditional_losses_15863
!conv2d_21/StatefulPartitionedCallStatefulPartitionedCall#dropout_28/PartitionedCall:output:0conv2d_21_16993conv2d_21_16995*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_21_layer_call_and_return_conditional_losses_15875
.batch_normalization_34/StatefulPartitionedCallStatefulPartitionedCall*conv2d_21/StatefulPartitionedCall:output:0batch_normalization_34_16998batch_normalization_34_17000batch_normalization_34_17002batch_normalization_34_17004*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_batch_normalization_34_layer_call_and_return_conditional_losses_15347ó
re_lu_29/PartitionedCallPartitionedCall7batch_normalization_34/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_re_lu_29_layer_call_and_return_conditional_losses_15895í
 max_pooling2d_21/PartitionedCallPartitionedCall!re_lu_29/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_max_pooling2d_21_layer_call_and_return_conditional_losses_15398é
dropout_29/PartitionedCallPartitionedCall)max_pooling2d_21/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_29_layer_call_and_return_conditional_losses_15903
!conv2d_22/StatefulPartitionedCallStatefulPartitionedCall#dropout_29/PartitionedCall:output:0conv2d_22_17010conv2d_22_17012*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_22_layer_call_and_return_conditional_losses_15915
.batch_normalization_35/StatefulPartitionedCallStatefulPartitionedCall*conv2d_22/StatefulPartitionedCall:output:0batch_normalization_35_17015batch_normalization_35_17017batch_normalization_35_17019batch_normalization_35_17021*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_batch_normalization_35_layer_call_and_return_conditional_losses_15423ó
re_lu_30/PartitionedCallPartitionedCall7batch_normalization_35/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_re_lu_30_layer_call_and_return_conditional_losses_15935í
 max_pooling2d_22/PartitionedCallPartitionedCall!re_lu_30/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_max_pooling2d_22_layer_call_and_return_conditional_losses_15474é
dropout_30/PartitionedCallPartitionedCall)max_pooling2d_22/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_30_layer_call_and_return_conditional_losses_15943
!conv2d_23/StatefulPartitionedCallStatefulPartitionedCall#dropout_30/PartitionedCall:output:0conv2d_23_17027conv2d_23_17029*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_23_layer_call_and_return_conditional_losses_15955
.batch_normalization_36/StatefulPartitionedCallStatefulPartitionedCall*conv2d_23/StatefulPartitionedCall:output:0batch_normalization_36_17032batch_normalization_36_17034batch_normalization_36_17036batch_normalization_36_17038*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_batch_normalization_36_layer_call_and_return_conditional_losses_15499ó
re_lu_31/PartitionedCallPartitionedCall7batch_normalization_36/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_re_lu_31_layer_call_and_return_conditional_losses_15975í
 max_pooling2d_23/PartitionedCallPartitionedCall!re_lu_31/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_max_pooling2d_23_layer_call_and_return_conditional_losses_15550é
dropout_31/PartitionedCallPartitionedCall)max_pooling2d_23/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_31_layer_call_and_return_conditional_losses_15983
!conv2d_24/StatefulPartitionedCallStatefulPartitionedCall#dropout_31/PartitionedCall:output:0conv2d_24_17044conv2d_24_17046*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_24_layer_call_and_return_conditional_losses_15995
.batch_normalization_37/StatefulPartitionedCallStatefulPartitionedCall*conv2d_24/StatefulPartitionedCall:output:0batch_normalization_37_17049batch_normalization_37_17051batch_normalization_37_17053batch_normalization_37_17055*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_batch_normalization_37_layer_call_and_return_conditional_losses_15575ô
re_lu_32/PartitionedCallPartitionedCall7batch_normalization_37/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_re_lu_32_layer_call_and_return_conditional_losses_16015î
 max_pooling2d_24/PartitionedCallPartitionedCall!re_lu_32/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_max_pooling2d_24_layer_call_and_return_conditional_losses_15626ê
dropout_32/PartitionedCallPartitionedCall)max_pooling2d_24/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_32_layer_call_and_return_conditional_losses_16023Ú
flatten_4/PartitionedCallPartitionedCall#dropout_32/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_flatten_4_layer_call_and_return_conditional_losses_16031
#features512/StatefulPartitionedCallStatefulPartitionedCall"flatten_4/PartitionedCall:output:0features512_17062features512_17064*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_features512_layer_call_and_return_conditional_losses_16043
.batch_normalization_38/StatefulPartitionedCallStatefulPartitionedCall,features512/StatefulPartitionedCall:output:0batch_normalization_38_17067batch_normalization_38_17069batch_normalization_38_17071batch_normalization_38_17073*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_batch_normalization_38_layer_call_and_return_conditional_losses_15653ì
re_lu_33/PartitionedCallPartitionedCall7batch_normalization_38/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_re_lu_33_layer_call_and_return_conditional_losses_16063Ú
dropout_33/PartitionedCallPartitionedCall!re_lu_33/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_33_layer_call_and_return_conditional_losses_16070
#features256/StatefulPartitionedCallStatefulPartitionedCall#dropout_33/PartitionedCall:output:0features256_17078features256_17080*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_features256_layer_call_and_return_conditional_losses_16082
.batch_normalization_39/StatefulPartitionedCallStatefulPartitionedCall,features256/StatefulPartitionedCall:output:0batch_normalization_39_17083batch_normalization_39_17085batch_normalization_39_17087batch_normalization_39_17089*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_batch_normalization_39_layer_call_and_return_conditional_losses_15735ì
re_lu_34/PartitionedCallPartitionedCall7batch_normalization_39/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_re_lu_34_layer_call_and_return_conditional_losses_16102Ú
dropout_34/PartitionedCallPartitionedCall!re_lu_34/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_34_layer_call_and_return_conditional_losses_16109
dense_4/StatefulPartitionedCallStatefulPartitionedCall#dropout_34/PartitionedCall:output:0dense_4_17094dense_4_17096*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_4_layer_call_and_return_conditional_losses_16122w
IdentityIdentity(dense_4/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿð
NoOpNoOp/^batch_normalization_32/StatefulPartitionedCall/^batch_normalization_33/StatefulPartitionedCall/^batch_normalization_34/StatefulPartitionedCall/^batch_normalization_35/StatefulPartitionedCall/^batch_normalization_36/StatefulPartitionedCall/^batch_normalization_37/StatefulPartitionedCall/^batch_normalization_38/StatefulPartitionedCall/^batch_normalization_39/StatefulPartitionedCall"^conv2d_20/StatefulPartitionedCall"^conv2d_21/StatefulPartitionedCall"^conv2d_22/StatefulPartitionedCall"^conv2d_23/StatefulPartitionedCall"^conv2d_24/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall$^features256/StatefulPartitionedCall$^features512/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapesy
w:ÿÿÿÿÿÿÿÿÿc(: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2`
.batch_normalization_32/StatefulPartitionedCall.batch_normalization_32/StatefulPartitionedCall2`
.batch_normalization_33/StatefulPartitionedCall.batch_normalization_33/StatefulPartitionedCall2`
.batch_normalization_34/StatefulPartitionedCall.batch_normalization_34/StatefulPartitionedCall2`
.batch_normalization_35/StatefulPartitionedCall.batch_normalization_35/StatefulPartitionedCall2`
.batch_normalization_36/StatefulPartitionedCall.batch_normalization_36/StatefulPartitionedCall2`
.batch_normalization_37/StatefulPartitionedCall.batch_normalization_37/StatefulPartitionedCall2`
.batch_normalization_38/StatefulPartitionedCall.batch_normalization_38/StatefulPartitionedCall2`
.batch_normalization_39/StatefulPartitionedCall.batch_normalization_39/StatefulPartitionedCall2F
!conv2d_20/StatefulPartitionedCall!conv2d_20/StatefulPartitionedCall2F
!conv2d_21/StatefulPartitionedCall!conv2d_21/StatefulPartitionedCall2F
!conv2d_22/StatefulPartitionedCall!conv2d_22/StatefulPartitionedCall2F
!conv2d_23/StatefulPartitionedCall!conv2d_23/StatefulPartitionedCall2F
!conv2d_24/StatefulPartitionedCall!conv2d_24/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2J
#features256/StatefulPartitionedCall#features256/StatefulPartitionedCall2J
#features512/StatefulPartitionedCall#features512/StatefulPartitionedCall:\ X
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿc(
)
_user_specified_namereshape_4_input

g
K__inference_max_pooling2d_20_layer_call_and_return_conditional_losses_18201

inputs
identity¢
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Â
F
*__inference_dropout_30_layer_call_fn_18462

inputs
identity»
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_30_layer_call_and_return_conditional_losses_15943h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ :W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
ø
c
E__inference_dropout_30_layer_call_and_return_conditional_losses_18472

inputs

identity_1V
IdentityIdentityinputs*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ c

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ :W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
Ü
c
E__inference_dropout_34_layer_call_and_return_conditional_losses_19011

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
³

d
E__inference_dropout_28_layer_call_and_return_conditional_losses_16498

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *«ªª?l
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  >®
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1w
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1q
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1a
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ1:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
 
_user_specified_nameinputs
³

d
E__inference_dropout_31_layer_call_and_return_conditional_losses_16381

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *«ªª?l
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  >®
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@w
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@q
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@a
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs

À
Q__inference_batch_normalization_32_layer_call_and_return_conditional_losses_15238

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity¢AssignNewValue¢AssignNewValue_1¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0Ö
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
exponential_avg_factor%
×#<Æ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(Ð
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÔ
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
à
´
Q__inference_batch_normalization_39_layer_call_and_return_conditional_losses_15735

inputs0
!batchnorm_readvariableop_resource:	4
%batchnorm_mul_readvariableop_resource:	2
#batchnorm_readvariableop_1_resource:	2
#batchnorm_readvariableop_2_resource:	
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOpw
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:x
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:Q
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype0u
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:d
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ{
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:*
dtype0s
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:{
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:*
dtype0s
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:s
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ë
_
C__inference_re_lu_33_layer_call_and_return_conditional_losses_18860

inputs
identityG
ReluReluinputs*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
»
L
0__inference_max_pooling2d_22_layer_call_fn_18452

inputs
identityÜ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_max_pooling2d_22_layer_call_and_return_conditional_losses_15474
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

À
Q__inference_batch_normalization_34_layer_call_and_return_conditional_losses_18309

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity¢AssignNewValue¢AssignNewValue_1¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0Ö
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
exponential_avg_factor%
×#<Æ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(Ð
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÔ
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ë
_
C__inference_re_lu_34_layer_call_and_return_conditional_losses_18996

inputs
identityG
ReluReluinputs*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
±
Õ
6__inference_batch_normalization_38_layer_call_fn_18796

inputs
unknown:	
	unknown_0:	
	unknown_1:	
	unknown_2:	
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_batch_normalization_38_layer_call_and_return_conditional_losses_15700p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
à
´
Q__inference_batch_normalization_38_layer_call_and_return_conditional_losses_18816

inputs0
!batchnorm_readvariableop_resource:	4
%batchnorm_mul_readvariableop_resource:	2
#batchnorm_readvariableop_1_resource:	2
#batchnorm_readvariableop_2_resource:	
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOpw
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:x
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:Q
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype0u
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:d
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ{
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:*
dtype0s
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:{
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:*
dtype0s
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:s
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¢

ô
B__inference_dense_4_layer_call_and_return_conditional_losses_16122

inputs1
matmul_readvariableop_resource:	-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ô	
ú
F__inference_features256_layer_call_and_return_conditional_losses_16082

inputs2
matmul_readvariableop_resource:
.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
IdentityIdentityBiasAdd:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

À
Q__inference_batch_normalization_33_layer_call_and_return_conditional_losses_15302

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity¢AssignNewValue¢AssignNewValue_1¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0Ö
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
exponential_avg_factor%
×#<Æ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(Ð
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÔ
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

À
Q__inference_batch_normalization_34_layer_call_and_return_conditional_losses_15378

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity¢AssignNewValue¢AssignNewValue_1¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0Ö
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
exponential_avg_factor%
×#<Æ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(Ð
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÔ
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¯%
î
Q__inference_batch_normalization_38_layer_call_and_return_conditional_losses_18850

inputs6
'assignmovingavg_readvariableop_resource:	8
)assignmovingavg_1_readvariableop_resource:	4
%batchnorm_mul_readvariableop_resource:	0
!batchnorm_readvariableop_resource:	
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	*
	keep_dims(e
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	*
	keep_dims(n
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:*
squeeze_dims
 t
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes	
:*
dtype0
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes	
:y
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:¬
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes	
:*
dtype0
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes	
:
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:´
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:r
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:Q
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype0u
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:d
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:w
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype0q
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:s
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿê
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
	
Ñ
6__inference_batch_normalization_33_layer_call_fn_18145

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_batch_normalization_33_layer_call_and_return_conditional_losses_15302
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ñ
 
)__inference_conv2d_24_layer_call_fn_18621

inputs"
unknown:@
	unknown_0:	
identity¢StatefulPartitionedCallå
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_24_layer_call_and_return_conditional_losses_15995x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ@: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Ç³
5
 __inference__wrapped_model_15185
reshape_4_inputI
;sequential_4_batch_normalization_32_readvariableop_resource:K
=sequential_4_batch_normalization_32_readvariableop_1_resource:Z
Lsequential_4_batch_normalization_32_fusedbatchnormv3_readvariableop_resource:\
Nsequential_4_batch_normalization_32_fusedbatchnormv3_readvariableop_1_resource:O
5sequential_4_conv2d_20_conv2d_readvariableop_resource:D
6sequential_4_conv2d_20_biasadd_readvariableop_resource:I
;sequential_4_batch_normalization_33_readvariableop_resource:K
=sequential_4_batch_normalization_33_readvariableop_1_resource:Z
Lsequential_4_batch_normalization_33_fusedbatchnormv3_readvariableop_resource:\
Nsequential_4_batch_normalization_33_fusedbatchnormv3_readvariableop_1_resource:O
5sequential_4_conv2d_21_conv2d_readvariableop_resource:D
6sequential_4_conv2d_21_biasadd_readvariableop_resource:I
;sequential_4_batch_normalization_34_readvariableop_resource:K
=sequential_4_batch_normalization_34_readvariableop_1_resource:Z
Lsequential_4_batch_normalization_34_fusedbatchnormv3_readvariableop_resource:\
Nsequential_4_batch_normalization_34_fusedbatchnormv3_readvariableop_1_resource:O
5sequential_4_conv2d_22_conv2d_readvariableop_resource: D
6sequential_4_conv2d_22_biasadd_readvariableop_resource: I
;sequential_4_batch_normalization_35_readvariableop_resource: K
=sequential_4_batch_normalization_35_readvariableop_1_resource: Z
Lsequential_4_batch_normalization_35_fusedbatchnormv3_readvariableop_resource: \
Nsequential_4_batch_normalization_35_fusedbatchnormv3_readvariableop_1_resource: O
5sequential_4_conv2d_23_conv2d_readvariableop_resource: @D
6sequential_4_conv2d_23_biasadd_readvariableop_resource:@I
;sequential_4_batch_normalization_36_readvariableop_resource:@K
=sequential_4_batch_normalization_36_readvariableop_1_resource:@Z
Lsequential_4_batch_normalization_36_fusedbatchnormv3_readvariableop_resource:@\
Nsequential_4_batch_normalization_36_fusedbatchnormv3_readvariableop_1_resource:@P
5sequential_4_conv2d_24_conv2d_readvariableop_resource:@E
6sequential_4_conv2d_24_biasadd_readvariableop_resource:	J
;sequential_4_batch_normalization_37_readvariableop_resource:	L
=sequential_4_batch_normalization_37_readvariableop_1_resource:	[
Lsequential_4_batch_normalization_37_fusedbatchnormv3_readvariableop_resource:	]
Nsequential_4_batch_normalization_37_fusedbatchnormv3_readvariableop_1_resource:	K
7sequential_4_features512_matmul_readvariableop_resource:
G
8sequential_4_features512_biasadd_readvariableop_resource:	T
Esequential_4_batch_normalization_38_batchnorm_readvariableop_resource:	X
Isequential_4_batch_normalization_38_batchnorm_mul_readvariableop_resource:	V
Gsequential_4_batch_normalization_38_batchnorm_readvariableop_1_resource:	V
Gsequential_4_batch_normalization_38_batchnorm_readvariableop_2_resource:	K
7sequential_4_features256_matmul_readvariableop_resource:
G
8sequential_4_features256_biasadd_readvariableop_resource:	T
Esequential_4_batch_normalization_39_batchnorm_readvariableop_resource:	X
Isequential_4_batch_normalization_39_batchnorm_mul_readvariableop_resource:	V
Gsequential_4_batch_normalization_39_batchnorm_readvariableop_1_resource:	V
Gsequential_4_batch_normalization_39_batchnorm_readvariableop_2_resource:	F
3sequential_4_dense_4_matmul_readvariableop_resource:	B
4sequential_4_dense_4_biasadd_readvariableop_resource:
identity¢Csequential_4/batch_normalization_32/FusedBatchNormV3/ReadVariableOp¢Esequential_4/batch_normalization_32/FusedBatchNormV3/ReadVariableOp_1¢2sequential_4/batch_normalization_32/ReadVariableOp¢4sequential_4/batch_normalization_32/ReadVariableOp_1¢Csequential_4/batch_normalization_33/FusedBatchNormV3/ReadVariableOp¢Esequential_4/batch_normalization_33/FusedBatchNormV3/ReadVariableOp_1¢2sequential_4/batch_normalization_33/ReadVariableOp¢4sequential_4/batch_normalization_33/ReadVariableOp_1¢Csequential_4/batch_normalization_34/FusedBatchNormV3/ReadVariableOp¢Esequential_4/batch_normalization_34/FusedBatchNormV3/ReadVariableOp_1¢2sequential_4/batch_normalization_34/ReadVariableOp¢4sequential_4/batch_normalization_34/ReadVariableOp_1¢Csequential_4/batch_normalization_35/FusedBatchNormV3/ReadVariableOp¢Esequential_4/batch_normalization_35/FusedBatchNormV3/ReadVariableOp_1¢2sequential_4/batch_normalization_35/ReadVariableOp¢4sequential_4/batch_normalization_35/ReadVariableOp_1¢Csequential_4/batch_normalization_36/FusedBatchNormV3/ReadVariableOp¢Esequential_4/batch_normalization_36/FusedBatchNormV3/ReadVariableOp_1¢2sequential_4/batch_normalization_36/ReadVariableOp¢4sequential_4/batch_normalization_36/ReadVariableOp_1¢Csequential_4/batch_normalization_37/FusedBatchNormV3/ReadVariableOp¢Esequential_4/batch_normalization_37/FusedBatchNormV3/ReadVariableOp_1¢2sequential_4/batch_normalization_37/ReadVariableOp¢4sequential_4/batch_normalization_37/ReadVariableOp_1¢<sequential_4/batch_normalization_38/batchnorm/ReadVariableOp¢>sequential_4/batch_normalization_38/batchnorm/ReadVariableOp_1¢>sequential_4/batch_normalization_38/batchnorm/ReadVariableOp_2¢@sequential_4/batch_normalization_38/batchnorm/mul/ReadVariableOp¢<sequential_4/batch_normalization_39/batchnorm/ReadVariableOp¢>sequential_4/batch_normalization_39/batchnorm/ReadVariableOp_1¢>sequential_4/batch_normalization_39/batchnorm/ReadVariableOp_2¢@sequential_4/batch_normalization_39/batchnorm/mul/ReadVariableOp¢-sequential_4/conv2d_20/BiasAdd/ReadVariableOp¢,sequential_4/conv2d_20/Conv2D/ReadVariableOp¢-sequential_4/conv2d_21/BiasAdd/ReadVariableOp¢,sequential_4/conv2d_21/Conv2D/ReadVariableOp¢-sequential_4/conv2d_22/BiasAdd/ReadVariableOp¢,sequential_4/conv2d_22/Conv2D/ReadVariableOp¢-sequential_4/conv2d_23/BiasAdd/ReadVariableOp¢,sequential_4/conv2d_23/Conv2D/ReadVariableOp¢-sequential_4/conv2d_24/BiasAdd/ReadVariableOp¢,sequential_4/conv2d_24/Conv2D/ReadVariableOp¢+sequential_4/dense_4/BiasAdd/ReadVariableOp¢*sequential_4/dense_4/MatMul/ReadVariableOp¢/sequential_4/features256/BiasAdd/ReadVariableOp¢.sequential_4/features256/MatMul/ReadVariableOp¢/sequential_4/features512/BiasAdd/ReadVariableOp¢.sequential_4/features512/MatMul/ReadVariableOp[
sequential_4/reshape_4/ShapeShapereshape_4_input*
T0*
_output_shapes
:t
*sequential_4/reshape_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: v
,sequential_4/reshape_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:v
,sequential_4/reshape_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ä
$sequential_4/reshape_4/strided_sliceStridedSlice%sequential_4/reshape_4/Shape:output:03sequential_4/reshape_4/strided_slice/stack:output:05sequential_4/reshape_4/strided_slice/stack_1:output:05sequential_4/reshape_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskh
&sequential_4/reshape_4/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :ch
&sequential_4/reshape_4/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :(h
&sequential_4/reshape_4/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :
$sequential_4/reshape_4/Reshape/shapePack-sequential_4/reshape_4/strided_slice:output:0/sequential_4/reshape_4/Reshape/shape/1:output:0/sequential_4/reshape_4/Reshape/shape/2:output:0/sequential_4/reshape_4/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:£
sequential_4/reshape_4/ReshapeReshapereshape_4_input-sequential_4/reshape_4/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿc(ª
2sequential_4/batch_normalization_32/ReadVariableOpReadVariableOp;sequential_4_batch_normalization_32_readvariableop_resource*
_output_shapes
:*
dtype0®
4sequential_4/batch_normalization_32/ReadVariableOp_1ReadVariableOp=sequential_4_batch_normalization_32_readvariableop_1_resource*
_output_shapes
:*
dtype0Ì
Csequential_4/batch_normalization_32/FusedBatchNormV3/ReadVariableOpReadVariableOpLsequential_4_batch_normalization_32_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0Ð
Esequential_4/batch_normalization_32/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpNsequential_4_batch_normalization_32_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0
4sequential_4/batch_normalization_32/FusedBatchNormV3FusedBatchNormV3'sequential_4/reshape_4/Reshape:output:0:sequential_4/batch_normalization_32/ReadVariableOp:value:0<sequential_4/batch_normalization_32/ReadVariableOp_1:value:0Ksequential_4/batch_normalization_32/FusedBatchNormV3/ReadVariableOp:value:0Msequential_4/batch_normalization_32/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿc(:::::*
epsilon%o:*
is_training( ª
,sequential_4/conv2d_20/Conv2D/ReadVariableOpReadVariableOp5sequential_4_conv2d_20_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0ù
sequential_4/conv2d_20/Conv2DConv2D8sequential_4/batch_normalization_32/FusedBatchNormV3:y:04sequential_4/conv2d_20/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿc(*
paddingSAME*
strides
 
-sequential_4/conv2d_20/BiasAdd/ReadVariableOpReadVariableOp6sequential_4_conv2d_20_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Â
sequential_4/conv2d_20/BiasAddBiasAdd&sequential_4/conv2d_20/Conv2D:output:05sequential_4/conv2d_20/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿc(ª
2sequential_4/batch_normalization_33/ReadVariableOpReadVariableOp;sequential_4_batch_normalization_33_readvariableop_resource*
_output_shapes
:*
dtype0®
4sequential_4/batch_normalization_33/ReadVariableOp_1ReadVariableOp=sequential_4_batch_normalization_33_readvariableop_1_resource*
_output_shapes
:*
dtype0Ì
Csequential_4/batch_normalization_33/FusedBatchNormV3/ReadVariableOpReadVariableOpLsequential_4_batch_normalization_33_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0Ð
Esequential_4/batch_normalization_33/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpNsequential_4_batch_normalization_33_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0
4sequential_4/batch_normalization_33/FusedBatchNormV3FusedBatchNormV3'sequential_4/conv2d_20/BiasAdd:output:0:sequential_4/batch_normalization_33/ReadVariableOp:value:0<sequential_4/batch_normalization_33/ReadVariableOp_1:value:0Ksequential_4/batch_normalization_33/FusedBatchNormV3/ReadVariableOp:value:0Msequential_4/batch_normalization_33/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿc(:::::*
epsilon%o:*
is_training( 
sequential_4/re_lu_28/ReluRelu8sequential_4/batch_normalization_33/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿc(Ç
%sequential_4/max_pooling2d_20/MaxPoolMaxPool(sequential_4/re_lu_28/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*
ksize
*
paddingVALID*
strides

 sequential_4/dropout_28/IdentityIdentity.sequential_4/max_pooling2d_20/MaxPool:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1ª
,sequential_4/conv2d_21/Conv2D/ReadVariableOpReadVariableOp5sequential_4_conv2d_21_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0ê
sequential_4/conv2d_21/Conv2DConv2D)sequential_4/dropout_28/Identity:output:04sequential_4/conv2d_21/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*
paddingSAME*
strides
 
-sequential_4/conv2d_21/BiasAdd/ReadVariableOpReadVariableOp6sequential_4_conv2d_21_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Â
sequential_4/conv2d_21/BiasAddBiasAdd&sequential_4/conv2d_21/Conv2D:output:05sequential_4/conv2d_21/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1ª
2sequential_4/batch_normalization_34/ReadVariableOpReadVariableOp;sequential_4_batch_normalization_34_readvariableop_resource*
_output_shapes
:*
dtype0®
4sequential_4/batch_normalization_34/ReadVariableOp_1ReadVariableOp=sequential_4_batch_normalization_34_readvariableop_1_resource*
_output_shapes
:*
dtype0Ì
Csequential_4/batch_normalization_34/FusedBatchNormV3/ReadVariableOpReadVariableOpLsequential_4_batch_normalization_34_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0Ð
Esequential_4/batch_normalization_34/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpNsequential_4_batch_normalization_34_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0
4sequential_4/batch_normalization_34/FusedBatchNormV3FusedBatchNormV3'sequential_4/conv2d_21/BiasAdd:output:0:sequential_4/batch_normalization_34/ReadVariableOp:value:0<sequential_4/batch_normalization_34/ReadVariableOp_1:value:0Ksequential_4/batch_normalization_34/FusedBatchNormV3/ReadVariableOp:value:0Msequential_4/batch_normalization_34/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ1:::::*
epsilon%o:*
is_training( 
sequential_4/re_lu_29/ReluRelu8sequential_4/batch_normalization_34/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1Ç
%sequential_4/max_pooling2d_21/MaxPoolMaxPool(sequential_4/re_lu_29/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
ksize
*
paddingVALID*
strides

 sequential_4/dropout_29/IdentityIdentity.sequential_4/max_pooling2d_21/MaxPool:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
ª
,sequential_4/conv2d_22/Conv2D/ReadVariableOpReadVariableOp5sequential_4_conv2d_22_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0ê
sequential_4/conv2d_22/Conv2DConv2D)sequential_4/dropout_29/Identity:output:04sequential_4/conv2d_22/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 *
paddingSAME*
strides
 
-sequential_4/conv2d_22/BiasAdd/ReadVariableOpReadVariableOp6sequential_4_conv2d_22_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0Â
sequential_4/conv2d_22/BiasAddBiasAdd&sequential_4/conv2d_22/Conv2D:output:05sequential_4/conv2d_22/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 ª
2sequential_4/batch_normalization_35/ReadVariableOpReadVariableOp;sequential_4_batch_normalization_35_readvariableop_resource*
_output_shapes
: *
dtype0®
4sequential_4/batch_normalization_35/ReadVariableOp_1ReadVariableOp=sequential_4_batch_normalization_35_readvariableop_1_resource*
_output_shapes
: *
dtype0Ì
Csequential_4/batch_normalization_35/FusedBatchNormV3/ReadVariableOpReadVariableOpLsequential_4_batch_normalization_35_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0Ð
Esequential_4/batch_normalization_35/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpNsequential_4_batch_normalization_35_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0
4sequential_4/batch_normalization_35/FusedBatchNormV3FusedBatchNormV3'sequential_4/conv2d_22/BiasAdd:output:0:sequential_4/batch_normalization_35/ReadVariableOp:value:0<sequential_4/batch_normalization_35/ReadVariableOp_1:value:0Ksequential_4/batch_normalization_35/FusedBatchNormV3/ReadVariableOp:value:0Msequential_4/batch_normalization_35/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ
 : : : : :*
epsilon%o:*
is_training( 
sequential_4/re_lu_30/ReluRelu8sequential_4/batch_normalization_35/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 Ç
%sequential_4/max_pooling2d_22/MaxPoolMaxPool(sequential_4/re_lu_30/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
ksize
*
paddingVALID*
strides

 sequential_4/dropout_30/IdentityIdentity.sequential_4/max_pooling2d_22/MaxPool:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ª
,sequential_4/conv2d_23/Conv2D/ReadVariableOpReadVariableOp5sequential_4_conv2d_23_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0ê
sequential_4/conv2d_23/Conv2DConv2D)sequential_4/dropout_30/Identity:output:04sequential_4/conv2d_23/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingSAME*
strides
 
-sequential_4/conv2d_23/BiasAdd/ReadVariableOpReadVariableOp6sequential_4_conv2d_23_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Â
sequential_4/conv2d_23/BiasAddBiasAdd&sequential_4/conv2d_23/Conv2D:output:05sequential_4/conv2d_23/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@ª
2sequential_4/batch_normalization_36/ReadVariableOpReadVariableOp;sequential_4_batch_normalization_36_readvariableop_resource*
_output_shapes
:@*
dtype0®
4sequential_4/batch_normalization_36/ReadVariableOp_1ReadVariableOp=sequential_4_batch_normalization_36_readvariableop_1_resource*
_output_shapes
:@*
dtype0Ì
Csequential_4/batch_normalization_36/FusedBatchNormV3/ReadVariableOpReadVariableOpLsequential_4_batch_normalization_36_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0Ð
Esequential_4/batch_normalization_36/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpNsequential_4_batch_normalization_36_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0
4sequential_4/batch_normalization_36/FusedBatchNormV3FusedBatchNormV3'sequential_4/conv2d_23/BiasAdd:output:0:sequential_4/batch_normalization_36/ReadVariableOp:value:0<sequential_4/batch_normalization_36/ReadVariableOp_1:value:0Ksequential_4/batch_normalization_36/FusedBatchNormV3/ReadVariableOp:value:0Msequential_4/batch_normalization_36/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
is_training( 
sequential_4/re_lu_31/ReluRelu8sequential_4/batch_normalization_36/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Ç
%sequential_4/max_pooling2d_23/MaxPoolMaxPool(sequential_4/re_lu_31/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
ksize
*
paddingVALID*
strides

 sequential_4/dropout_31/IdentityIdentity.sequential_4/max_pooling2d_23/MaxPool:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@«
,sequential_4/conv2d_24/Conv2D/ReadVariableOpReadVariableOp5sequential_4_conv2d_24_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype0ë
sequential_4/conv2d_24/Conv2DConv2D)sequential_4/dropout_31/Identity:output:04sequential_4/conv2d_24/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
¡
-sequential_4/conv2d_24/BiasAdd/ReadVariableOpReadVariableOp6sequential_4_conv2d_24_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Ã
sequential_4/conv2d_24/BiasAddBiasAdd&sequential_4/conv2d_24/Conv2D:output:05sequential_4/conv2d_24/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ«
2sequential_4/batch_normalization_37/ReadVariableOpReadVariableOp;sequential_4_batch_normalization_37_readvariableop_resource*
_output_shapes	
:*
dtype0¯
4sequential_4/batch_normalization_37/ReadVariableOp_1ReadVariableOp=sequential_4_batch_normalization_37_readvariableop_1_resource*
_output_shapes	
:*
dtype0Í
Csequential_4/batch_normalization_37/FusedBatchNormV3/ReadVariableOpReadVariableOpLsequential_4_batch_normalization_37_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0Ñ
Esequential_4/batch_normalization_37/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpNsequential_4_batch_normalization_37_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0
4sequential_4/batch_normalization_37/FusedBatchNormV3FusedBatchNormV3'sequential_4/conv2d_24/BiasAdd:output:0:sequential_4/batch_normalization_37/ReadVariableOp:value:0<sequential_4/batch_normalization_37/ReadVariableOp_1:value:0Ksequential_4/batch_normalization_37/FusedBatchNormV3/ReadVariableOp:value:0Msequential_4/batch_normalization_37/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( 
sequential_4/re_lu_32/ReluRelu8sequential_4/batch_normalization_37/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
%sequential_4/max_pooling2d_24/MaxPoolMaxPool(sequential_4/re_lu_32/Relu:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides

 sequential_4/dropout_32/IdentityIdentity.sequential_4/max_pooling2d_24/MaxPool:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿm
sequential_4/flatten_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ  ®
sequential_4/flatten_4/ReshapeReshape)sequential_4/dropout_32/Identity:output:0%sequential_4/flatten_4/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨
.sequential_4/features512/MatMul/ReadVariableOpReadVariableOp7sequential_4_features512_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0½
sequential_4/features512/MatMulMatMul'sequential_4/flatten_4/Reshape:output:06sequential_4/features512/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥
/sequential_4/features512/BiasAdd/ReadVariableOpReadVariableOp8sequential_4_features512_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Â
 sequential_4/features512/BiasAddBiasAdd)sequential_4/features512/MatMul:product:07sequential_4/features512/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¿
<sequential_4/batch_normalization_38/batchnorm/ReadVariableOpReadVariableOpEsequential_4_batch_normalization_38_batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype0x
3sequential_4/batch_normalization_38/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:ä
1sequential_4/batch_normalization_38/batchnorm/addAddV2Dsequential_4/batch_normalization_38/batchnorm/ReadVariableOp:value:0<sequential_4/batch_normalization_38/batchnorm/add/y:output:0*
T0*
_output_shapes	
:
3sequential_4/batch_normalization_38/batchnorm/RsqrtRsqrt5sequential_4/batch_normalization_38/batchnorm/add:z:0*
T0*
_output_shapes	
:Ç
@sequential_4/batch_normalization_38/batchnorm/mul/ReadVariableOpReadVariableOpIsequential_4_batch_normalization_38_batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype0á
1sequential_4/batch_normalization_38/batchnorm/mulMul7sequential_4/batch_normalization_38/batchnorm/Rsqrt:y:0Hsequential_4/batch_normalization_38/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:Ï
3sequential_4/batch_normalization_38/batchnorm/mul_1Mul)sequential_4/features512/BiasAdd:output:05sequential_4/batch_normalization_38/batchnorm/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÃ
>sequential_4/batch_normalization_38/batchnorm/ReadVariableOp_1ReadVariableOpGsequential_4_batch_normalization_38_batchnorm_readvariableop_1_resource*
_output_shapes	
:*
dtype0ß
3sequential_4/batch_normalization_38/batchnorm/mul_2MulFsequential_4/batch_normalization_38/batchnorm/ReadVariableOp_1:value:05sequential_4/batch_normalization_38/batchnorm/mul:z:0*
T0*
_output_shapes	
:Ã
>sequential_4/batch_normalization_38/batchnorm/ReadVariableOp_2ReadVariableOpGsequential_4_batch_normalization_38_batchnorm_readvariableop_2_resource*
_output_shapes	
:*
dtype0ß
1sequential_4/batch_normalization_38/batchnorm/subSubFsequential_4/batch_normalization_38/batchnorm/ReadVariableOp_2:value:07sequential_4/batch_normalization_38/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:ß
3sequential_4/batch_normalization_38/batchnorm/add_1AddV27sequential_4/batch_normalization_38/batchnorm/mul_1:z:05sequential_4/batch_normalization_38/batchnorm/sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
sequential_4/re_lu_33/ReluRelu7sequential_4/batch_normalization_38/batchnorm/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 sequential_4/dropout_33/IdentityIdentity(sequential_4/re_lu_33/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨
.sequential_4/features256/MatMul/ReadVariableOpReadVariableOp7sequential_4_features256_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0¿
sequential_4/features256/MatMulMatMul)sequential_4/dropout_33/Identity:output:06sequential_4/features256/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥
/sequential_4/features256/BiasAdd/ReadVariableOpReadVariableOp8sequential_4_features256_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Â
 sequential_4/features256/BiasAddBiasAdd)sequential_4/features256/MatMul:product:07sequential_4/features256/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¿
<sequential_4/batch_normalization_39/batchnorm/ReadVariableOpReadVariableOpEsequential_4_batch_normalization_39_batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype0x
3sequential_4/batch_normalization_39/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:ä
1sequential_4/batch_normalization_39/batchnorm/addAddV2Dsequential_4/batch_normalization_39/batchnorm/ReadVariableOp:value:0<sequential_4/batch_normalization_39/batchnorm/add/y:output:0*
T0*
_output_shapes	
:
3sequential_4/batch_normalization_39/batchnorm/RsqrtRsqrt5sequential_4/batch_normalization_39/batchnorm/add:z:0*
T0*
_output_shapes	
:Ç
@sequential_4/batch_normalization_39/batchnorm/mul/ReadVariableOpReadVariableOpIsequential_4_batch_normalization_39_batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype0á
1sequential_4/batch_normalization_39/batchnorm/mulMul7sequential_4/batch_normalization_39/batchnorm/Rsqrt:y:0Hsequential_4/batch_normalization_39/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:Ï
3sequential_4/batch_normalization_39/batchnorm/mul_1Mul)sequential_4/features256/BiasAdd:output:05sequential_4/batch_normalization_39/batchnorm/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÃ
>sequential_4/batch_normalization_39/batchnorm/ReadVariableOp_1ReadVariableOpGsequential_4_batch_normalization_39_batchnorm_readvariableop_1_resource*
_output_shapes	
:*
dtype0ß
3sequential_4/batch_normalization_39/batchnorm/mul_2MulFsequential_4/batch_normalization_39/batchnorm/ReadVariableOp_1:value:05sequential_4/batch_normalization_39/batchnorm/mul:z:0*
T0*
_output_shapes	
:Ã
>sequential_4/batch_normalization_39/batchnorm/ReadVariableOp_2ReadVariableOpGsequential_4_batch_normalization_39_batchnorm_readvariableop_2_resource*
_output_shapes	
:*
dtype0ß
1sequential_4/batch_normalization_39/batchnorm/subSubFsequential_4/batch_normalization_39/batchnorm/ReadVariableOp_2:value:07sequential_4/batch_normalization_39/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:ß
3sequential_4/batch_normalization_39/batchnorm/add_1AddV27sequential_4/batch_normalization_39/batchnorm/mul_1:z:05sequential_4/batch_normalization_39/batchnorm/sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
sequential_4/re_lu_34/ReluRelu7sequential_4/batch_normalization_39/batchnorm/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 sequential_4/dropout_34/IdentityIdentity(sequential_4/re_lu_34/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*sequential_4/dense_4/MatMul/ReadVariableOpReadVariableOp3sequential_4_dense_4_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0¶
sequential_4/dense_4/MatMulMatMul)sequential_4/dropout_34/Identity:output:02sequential_4/dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
+sequential_4/dense_4/BiasAdd/ReadVariableOpReadVariableOp4sequential_4_dense_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0µ
sequential_4/dense_4/BiasAddBiasAdd%sequential_4/dense_4/MatMul:product:03sequential_4/dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
sequential_4/dense_4/SoftmaxSoftmax%sequential_4/dense_4/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿu
IdentityIdentity&sequential_4/dense_4/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¦
NoOpNoOpD^sequential_4/batch_normalization_32/FusedBatchNormV3/ReadVariableOpF^sequential_4/batch_normalization_32/FusedBatchNormV3/ReadVariableOp_13^sequential_4/batch_normalization_32/ReadVariableOp5^sequential_4/batch_normalization_32/ReadVariableOp_1D^sequential_4/batch_normalization_33/FusedBatchNormV3/ReadVariableOpF^sequential_4/batch_normalization_33/FusedBatchNormV3/ReadVariableOp_13^sequential_4/batch_normalization_33/ReadVariableOp5^sequential_4/batch_normalization_33/ReadVariableOp_1D^sequential_4/batch_normalization_34/FusedBatchNormV3/ReadVariableOpF^sequential_4/batch_normalization_34/FusedBatchNormV3/ReadVariableOp_13^sequential_4/batch_normalization_34/ReadVariableOp5^sequential_4/batch_normalization_34/ReadVariableOp_1D^sequential_4/batch_normalization_35/FusedBatchNormV3/ReadVariableOpF^sequential_4/batch_normalization_35/FusedBatchNormV3/ReadVariableOp_13^sequential_4/batch_normalization_35/ReadVariableOp5^sequential_4/batch_normalization_35/ReadVariableOp_1D^sequential_4/batch_normalization_36/FusedBatchNormV3/ReadVariableOpF^sequential_4/batch_normalization_36/FusedBatchNormV3/ReadVariableOp_13^sequential_4/batch_normalization_36/ReadVariableOp5^sequential_4/batch_normalization_36/ReadVariableOp_1D^sequential_4/batch_normalization_37/FusedBatchNormV3/ReadVariableOpF^sequential_4/batch_normalization_37/FusedBatchNormV3/ReadVariableOp_13^sequential_4/batch_normalization_37/ReadVariableOp5^sequential_4/batch_normalization_37/ReadVariableOp_1=^sequential_4/batch_normalization_38/batchnorm/ReadVariableOp?^sequential_4/batch_normalization_38/batchnorm/ReadVariableOp_1?^sequential_4/batch_normalization_38/batchnorm/ReadVariableOp_2A^sequential_4/batch_normalization_38/batchnorm/mul/ReadVariableOp=^sequential_4/batch_normalization_39/batchnorm/ReadVariableOp?^sequential_4/batch_normalization_39/batchnorm/ReadVariableOp_1?^sequential_4/batch_normalization_39/batchnorm/ReadVariableOp_2A^sequential_4/batch_normalization_39/batchnorm/mul/ReadVariableOp.^sequential_4/conv2d_20/BiasAdd/ReadVariableOp-^sequential_4/conv2d_20/Conv2D/ReadVariableOp.^sequential_4/conv2d_21/BiasAdd/ReadVariableOp-^sequential_4/conv2d_21/Conv2D/ReadVariableOp.^sequential_4/conv2d_22/BiasAdd/ReadVariableOp-^sequential_4/conv2d_22/Conv2D/ReadVariableOp.^sequential_4/conv2d_23/BiasAdd/ReadVariableOp-^sequential_4/conv2d_23/Conv2D/ReadVariableOp.^sequential_4/conv2d_24/BiasAdd/ReadVariableOp-^sequential_4/conv2d_24/Conv2D/ReadVariableOp,^sequential_4/dense_4/BiasAdd/ReadVariableOp+^sequential_4/dense_4/MatMul/ReadVariableOp0^sequential_4/features256/BiasAdd/ReadVariableOp/^sequential_4/features256/MatMul/ReadVariableOp0^sequential_4/features512/BiasAdd/ReadVariableOp/^sequential_4/features512/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapesy
w:ÿÿÿÿÿÿÿÿÿc(: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2
Csequential_4/batch_normalization_32/FusedBatchNormV3/ReadVariableOpCsequential_4/batch_normalization_32/FusedBatchNormV3/ReadVariableOp2
Esequential_4/batch_normalization_32/FusedBatchNormV3/ReadVariableOp_1Esequential_4/batch_normalization_32/FusedBatchNormV3/ReadVariableOp_12h
2sequential_4/batch_normalization_32/ReadVariableOp2sequential_4/batch_normalization_32/ReadVariableOp2l
4sequential_4/batch_normalization_32/ReadVariableOp_14sequential_4/batch_normalization_32/ReadVariableOp_12
Csequential_4/batch_normalization_33/FusedBatchNormV3/ReadVariableOpCsequential_4/batch_normalization_33/FusedBatchNormV3/ReadVariableOp2
Esequential_4/batch_normalization_33/FusedBatchNormV3/ReadVariableOp_1Esequential_4/batch_normalization_33/FusedBatchNormV3/ReadVariableOp_12h
2sequential_4/batch_normalization_33/ReadVariableOp2sequential_4/batch_normalization_33/ReadVariableOp2l
4sequential_4/batch_normalization_33/ReadVariableOp_14sequential_4/batch_normalization_33/ReadVariableOp_12
Csequential_4/batch_normalization_34/FusedBatchNormV3/ReadVariableOpCsequential_4/batch_normalization_34/FusedBatchNormV3/ReadVariableOp2
Esequential_4/batch_normalization_34/FusedBatchNormV3/ReadVariableOp_1Esequential_4/batch_normalization_34/FusedBatchNormV3/ReadVariableOp_12h
2sequential_4/batch_normalization_34/ReadVariableOp2sequential_4/batch_normalization_34/ReadVariableOp2l
4sequential_4/batch_normalization_34/ReadVariableOp_14sequential_4/batch_normalization_34/ReadVariableOp_12
Csequential_4/batch_normalization_35/FusedBatchNormV3/ReadVariableOpCsequential_4/batch_normalization_35/FusedBatchNormV3/ReadVariableOp2
Esequential_4/batch_normalization_35/FusedBatchNormV3/ReadVariableOp_1Esequential_4/batch_normalization_35/FusedBatchNormV3/ReadVariableOp_12h
2sequential_4/batch_normalization_35/ReadVariableOp2sequential_4/batch_normalization_35/ReadVariableOp2l
4sequential_4/batch_normalization_35/ReadVariableOp_14sequential_4/batch_normalization_35/ReadVariableOp_12
Csequential_4/batch_normalization_36/FusedBatchNormV3/ReadVariableOpCsequential_4/batch_normalization_36/FusedBatchNormV3/ReadVariableOp2
Esequential_4/batch_normalization_36/FusedBatchNormV3/ReadVariableOp_1Esequential_4/batch_normalization_36/FusedBatchNormV3/ReadVariableOp_12h
2sequential_4/batch_normalization_36/ReadVariableOp2sequential_4/batch_normalization_36/ReadVariableOp2l
4sequential_4/batch_normalization_36/ReadVariableOp_14sequential_4/batch_normalization_36/ReadVariableOp_12
Csequential_4/batch_normalization_37/FusedBatchNormV3/ReadVariableOpCsequential_4/batch_normalization_37/FusedBatchNormV3/ReadVariableOp2
Esequential_4/batch_normalization_37/FusedBatchNormV3/ReadVariableOp_1Esequential_4/batch_normalization_37/FusedBatchNormV3/ReadVariableOp_12h
2sequential_4/batch_normalization_37/ReadVariableOp2sequential_4/batch_normalization_37/ReadVariableOp2l
4sequential_4/batch_normalization_37/ReadVariableOp_14sequential_4/batch_normalization_37/ReadVariableOp_12|
<sequential_4/batch_normalization_38/batchnorm/ReadVariableOp<sequential_4/batch_normalization_38/batchnorm/ReadVariableOp2
>sequential_4/batch_normalization_38/batchnorm/ReadVariableOp_1>sequential_4/batch_normalization_38/batchnorm/ReadVariableOp_12
>sequential_4/batch_normalization_38/batchnorm/ReadVariableOp_2>sequential_4/batch_normalization_38/batchnorm/ReadVariableOp_22
@sequential_4/batch_normalization_38/batchnorm/mul/ReadVariableOp@sequential_4/batch_normalization_38/batchnorm/mul/ReadVariableOp2|
<sequential_4/batch_normalization_39/batchnorm/ReadVariableOp<sequential_4/batch_normalization_39/batchnorm/ReadVariableOp2
>sequential_4/batch_normalization_39/batchnorm/ReadVariableOp_1>sequential_4/batch_normalization_39/batchnorm/ReadVariableOp_12
>sequential_4/batch_normalization_39/batchnorm/ReadVariableOp_2>sequential_4/batch_normalization_39/batchnorm/ReadVariableOp_22
@sequential_4/batch_normalization_39/batchnorm/mul/ReadVariableOp@sequential_4/batch_normalization_39/batchnorm/mul/ReadVariableOp2^
-sequential_4/conv2d_20/BiasAdd/ReadVariableOp-sequential_4/conv2d_20/BiasAdd/ReadVariableOp2\
,sequential_4/conv2d_20/Conv2D/ReadVariableOp,sequential_4/conv2d_20/Conv2D/ReadVariableOp2^
-sequential_4/conv2d_21/BiasAdd/ReadVariableOp-sequential_4/conv2d_21/BiasAdd/ReadVariableOp2\
,sequential_4/conv2d_21/Conv2D/ReadVariableOp,sequential_4/conv2d_21/Conv2D/ReadVariableOp2^
-sequential_4/conv2d_22/BiasAdd/ReadVariableOp-sequential_4/conv2d_22/BiasAdd/ReadVariableOp2\
,sequential_4/conv2d_22/Conv2D/ReadVariableOp,sequential_4/conv2d_22/Conv2D/ReadVariableOp2^
-sequential_4/conv2d_23/BiasAdd/ReadVariableOp-sequential_4/conv2d_23/BiasAdd/ReadVariableOp2\
,sequential_4/conv2d_23/Conv2D/ReadVariableOp,sequential_4/conv2d_23/Conv2D/ReadVariableOp2^
-sequential_4/conv2d_24/BiasAdd/ReadVariableOp-sequential_4/conv2d_24/BiasAdd/ReadVariableOp2\
,sequential_4/conv2d_24/Conv2D/ReadVariableOp,sequential_4/conv2d_24/Conv2D/ReadVariableOp2Z
+sequential_4/dense_4/BiasAdd/ReadVariableOp+sequential_4/dense_4/BiasAdd/ReadVariableOp2X
*sequential_4/dense_4/MatMul/ReadVariableOp*sequential_4/dense_4/MatMul/ReadVariableOp2b
/sequential_4/features256/BiasAdd/ReadVariableOp/sequential_4/features256/BiasAdd/ReadVariableOp2`
.sequential_4/features256/MatMul/ReadVariableOp.sequential_4/features256/MatMul/ReadVariableOp2b
/sequential_4/features512/BiasAdd/ReadVariableOp/sequential_4/features512/BiasAdd/ReadVariableOp2`
.sequential_4/features512/MatMul/ReadVariableOp.sequential_4/features512/MatMul/ReadVariableOp:\ X
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿc(
)
_user_specified_namereshape_4_input
Â
F
*__inference_dropout_31_layer_call_fn_18590

inputs
identity»
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_31_layer_call_and_return_conditional_losses_15983h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
¢
D
(__inference_re_lu_33_layer_call_fn_18855

inputs
identity²
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_re_lu_33_layer_call_and_return_conditional_losses_16063a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
±c

__inference__traced_save_19210
file_prefix;
7savev2_batch_normalization_32_gamma_read_readvariableop:
6savev2_batch_normalization_32_beta_read_readvariableopA
=savev2_batch_normalization_32_moving_mean_read_readvariableopE
Asavev2_batch_normalization_32_moving_variance_read_readvariableop/
+savev2_conv2d_20_kernel_read_readvariableop-
)savev2_conv2d_20_bias_read_readvariableop;
7savev2_batch_normalization_33_gamma_read_readvariableop:
6savev2_batch_normalization_33_beta_read_readvariableopA
=savev2_batch_normalization_33_moving_mean_read_readvariableopE
Asavev2_batch_normalization_33_moving_variance_read_readvariableop/
+savev2_conv2d_21_kernel_read_readvariableop-
)savev2_conv2d_21_bias_read_readvariableop;
7savev2_batch_normalization_34_gamma_read_readvariableop:
6savev2_batch_normalization_34_beta_read_readvariableopA
=savev2_batch_normalization_34_moving_mean_read_readvariableopE
Asavev2_batch_normalization_34_moving_variance_read_readvariableop/
+savev2_conv2d_22_kernel_read_readvariableop-
)savev2_conv2d_22_bias_read_readvariableop;
7savev2_batch_normalization_35_gamma_read_readvariableop:
6savev2_batch_normalization_35_beta_read_readvariableopA
=savev2_batch_normalization_35_moving_mean_read_readvariableopE
Asavev2_batch_normalization_35_moving_variance_read_readvariableop/
+savev2_conv2d_23_kernel_read_readvariableop-
)savev2_conv2d_23_bias_read_readvariableop;
7savev2_batch_normalization_36_gamma_read_readvariableop:
6savev2_batch_normalization_36_beta_read_readvariableopA
=savev2_batch_normalization_36_moving_mean_read_readvariableopE
Asavev2_batch_normalization_36_moving_variance_read_readvariableop/
+savev2_conv2d_24_kernel_read_readvariableop-
)savev2_conv2d_24_bias_read_readvariableop;
7savev2_batch_normalization_37_gamma_read_readvariableop:
6savev2_batch_normalization_37_beta_read_readvariableopA
=savev2_batch_normalization_37_moving_mean_read_readvariableopE
Asavev2_batch_normalization_37_moving_variance_read_readvariableop1
-savev2_features512_kernel_read_readvariableop/
+savev2_features512_bias_read_readvariableop;
7savev2_batch_normalization_38_gamma_read_readvariableop:
6savev2_batch_normalization_38_beta_read_readvariableopA
=savev2_batch_normalization_38_moving_mean_read_readvariableopE
Asavev2_batch_normalization_38_moving_variance_read_readvariableop1
-savev2_features256_kernel_read_readvariableop/
+savev2_features256_bias_read_readvariableop;
7savev2_batch_normalization_39_gamma_read_readvariableop:
6savev2_batch_normalization_39_beta_read_readvariableopA
=savev2_batch_normalization_39_moving_mean_read_readvariableopE
Asavev2_batch_normalization_39_moving_variance_read_readvariableop-
)savev2_dense_4_kernel_read_readvariableop+
'savev2_dense_4_bias_read_readvariableop
savev2_const

identity_1¢MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ä
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:1*
dtype0*
valueB1B5layer_with_weights-0/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-0/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-0/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-2/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-2/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-2/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-4/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-4/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-6/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-6/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-6/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-8/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-8/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-8/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-10/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-10/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-10/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-12/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-12/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-12/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-14/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-14/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-14/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-14/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-15/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-15/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHÏ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:1*
dtype0*u
valuelBj1B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ¼
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:07savev2_batch_normalization_32_gamma_read_readvariableop6savev2_batch_normalization_32_beta_read_readvariableop=savev2_batch_normalization_32_moving_mean_read_readvariableopAsavev2_batch_normalization_32_moving_variance_read_readvariableop+savev2_conv2d_20_kernel_read_readvariableop)savev2_conv2d_20_bias_read_readvariableop7savev2_batch_normalization_33_gamma_read_readvariableop6savev2_batch_normalization_33_beta_read_readvariableop=savev2_batch_normalization_33_moving_mean_read_readvariableopAsavev2_batch_normalization_33_moving_variance_read_readvariableop+savev2_conv2d_21_kernel_read_readvariableop)savev2_conv2d_21_bias_read_readvariableop7savev2_batch_normalization_34_gamma_read_readvariableop6savev2_batch_normalization_34_beta_read_readvariableop=savev2_batch_normalization_34_moving_mean_read_readvariableopAsavev2_batch_normalization_34_moving_variance_read_readvariableop+savev2_conv2d_22_kernel_read_readvariableop)savev2_conv2d_22_bias_read_readvariableop7savev2_batch_normalization_35_gamma_read_readvariableop6savev2_batch_normalization_35_beta_read_readvariableop=savev2_batch_normalization_35_moving_mean_read_readvariableopAsavev2_batch_normalization_35_moving_variance_read_readvariableop+savev2_conv2d_23_kernel_read_readvariableop)savev2_conv2d_23_bias_read_readvariableop7savev2_batch_normalization_36_gamma_read_readvariableop6savev2_batch_normalization_36_beta_read_readvariableop=savev2_batch_normalization_36_moving_mean_read_readvariableopAsavev2_batch_normalization_36_moving_variance_read_readvariableop+savev2_conv2d_24_kernel_read_readvariableop)savev2_conv2d_24_bias_read_readvariableop7savev2_batch_normalization_37_gamma_read_readvariableop6savev2_batch_normalization_37_beta_read_readvariableop=savev2_batch_normalization_37_moving_mean_read_readvariableopAsavev2_batch_normalization_37_moving_variance_read_readvariableop-savev2_features512_kernel_read_readvariableop+savev2_features512_bias_read_readvariableop7savev2_batch_normalization_38_gamma_read_readvariableop6savev2_batch_normalization_38_beta_read_readvariableop=savev2_batch_normalization_38_moving_mean_read_readvariableopAsavev2_batch_normalization_38_moving_variance_read_readvariableop-savev2_features256_kernel_read_readvariableop+savev2_features256_bias_read_readvariableop7savev2_batch_normalization_39_gamma_read_readvariableop6savev2_batch_normalization_39_beta_read_readvariableop=savev2_batch_normalization_39_moving_mean_read_readvariableopAsavev2_batch_normalization_39_moving_variance_read_readvariableop)savev2_dense_4_kernel_read_readvariableop'savev2_dense_4_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *?
dtypes5
321
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*
_input_shapes
: ::::::::::::::::: : : : : : : @:@:@:@:@:@:@::::::
::::::
::::::	:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 	

_output_shapes
:: 


_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
: @: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@:-)
'
_output_shapes
:@:!

_output_shapes	
::!

_output_shapes	
::! 

_output_shapes	
::!!

_output_shapes	
::!"

_output_shapes	
::&#"
 
_output_shapes
:
:!$

_output_shapes	
::!%

_output_shapes	
::!&

_output_shapes	
::!'

_output_shapes	
::!(

_output_shapes	
::&)"
 
_output_shapes
:
:!*

_output_shapes	
::!+

_output_shapes	
::!,

_output_shapes	
::!-

_output_shapes	
::!.

_output_shapes	
::%/!

_output_shapes
:	: 0

_output_shapes
::1

_output_shapes
: 
û	
d
E__inference_dropout_33_layer_call_and_return_conditional_losses_16297

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *«ªª?e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿC
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  >§
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿp
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿj
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ì

Q__inference_batch_normalization_32_layer_call_and_return_conditional_losses_15207

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0È
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ°
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ø
c
*__inference_dropout_34_layer_call_fn_19006

inputs
identity¢StatefulPartitionedCallÄ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_34_layer_call_and_return_conditional_losses_16258p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
	
Õ
6__inference_batch_normalization_37_layer_call_fn_18657

inputs
unknown:	
	unknown_0:	
	unknown_1:	
	unknown_2:	
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_batch_normalization_37_layer_call_and_return_conditional_losses_15606
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
í

)__inference_conv2d_21_layer_call_fn_18237

inputs!
unknown:
	unknown_0:
identity¢StatefulPartitionedCallä
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_21_layer_call_and_return_conditional_losses_15875w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ1: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
 
_user_specified_nameinputs
¢

ô
B__inference_dense_4_layer_call_and_return_conditional_losses_19043

inputs1
matmul_readvariableop_resource:	-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Æ
F
*__inference_dropout_32_layer_call_fn_18718

inputs
identity¼
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_32_layer_call_and_return_conditional_losses_16023i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
üÁ
Ù 
!__inference__traced_restore_19364
file_prefix;
-assignvariableop_batch_normalization_32_gamma:<
.assignvariableop_1_batch_normalization_32_beta:C
5assignvariableop_2_batch_normalization_32_moving_mean:G
9assignvariableop_3_batch_normalization_32_moving_variance:=
#assignvariableop_4_conv2d_20_kernel:/
!assignvariableop_5_conv2d_20_bias:=
/assignvariableop_6_batch_normalization_33_gamma:<
.assignvariableop_7_batch_normalization_33_beta:C
5assignvariableop_8_batch_normalization_33_moving_mean:G
9assignvariableop_9_batch_normalization_33_moving_variance:>
$assignvariableop_10_conv2d_21_kernel:0
"assignvariableop_11_conv2d_21_bias:>
0assignvariableop_12_batch_normalization_34_gamma:=
/assignvariableop_13_batch_normalization_34_beta:D
6assignvariableop_14_batch_normalization_34_moving_mean:H
:assignvariableop_15_batch_normalization_34_moving_variance:>
$assignvariableop_16_conv2d_22_kernel: 0
"assignvariableop_17_conv2d_22_bias: >
0assignvariableop_18_batch_normalization_35_gamma: =
/assignvariableop_19_batch_normalization_35_beta: D
6assignvariableop_20_batch_normalization_35_moving_mean: H
:assignvariableop_21_batch_normalization_35_moving_variance: >
$assignvariableop_22_conv2d_23_kernel: @0
"assignvariableop_23_conv2d_23_bias:@>
0assignvariableop_24_batch_normalization_36_gamma:@=
/assignvariableop_25_batch_normalization_36_beta:@D
6assignvariableop_26_batch_normalization_36_moving_mean:@H
:assignvariableop_27_batch_normalization_36_moving_variance:@?
$assignvariableop_28_conv2d_24_kernel:@1
"assignvariableop_29_conv2d_24_bias:	?
0assignvariableop_30_batch_normalization_37_gamma:	>
/assignvariableop_31_batch_normalization_37_beta:	E
6assignvariableop_32_batch_normalization_37_moving_mean:	I
:assignvariableop_33_batch_normalization_37_moving_variance:	:
&assignvariableop_34_features512_kernel:
3
$assignvariableop_35_features512_bias:	?
0assignvariableop_36_batch_normalization_38_gamma:	>
/assignvariableop_37_batch_normalization_38_beta:	E
6assignvariableop_38_batch_normalization_38_moving_mean:	I
:assignvariableop_39_batch_normalization_38_moving_variance:	:
&assignvariableop_40_features256_kernel:
3
$assignvariableop_41_features256_bias:	?
0assignvariableop_42_batch_normalization_39_gamma:	>
/assignvariableop_43_batch_normalization_39_beta:	E
6assignvariableop_44_batch_normalization_39_moving_mean:	I
:assignvariableop_45_batch_normalization_39_moving_variance:	5
"assignvariableop_46_dense_4_kernel:	.
 assignvariableop_47_dense_4_bias:
identity_49¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_22¢AssignVariableOp_23¢AssignVariableOp_24¢AssignVariableOp_25¢AssignVariableOp_26¢AssignVariableOp_27¢AssignVariableOp_28¢AssignVariableOp_29¢AssignVariableOp_3¢AssignVariableOp_30¢AssignVariableOp_31¢AssignVariableOp_32¢AssignVariableOp_33¢AssignVariableOp_34¢AssignVariableOp_35¢AssignVariableOp_36¢AssignVariableOp_37¢AssignVariableOp_38¢AssignVariableOp_39¢AssignVariableOp_4¢AssignVariableOp_40¢AssignVariableOp_41¢AssignVariableOp_42¢AssignVariableOp_43¢AssignVariableOp_44¢AssignVariableOp_45¢AssignVariableOp_46¢AssignVariableOp_47¢AssignVariableOp_5¢AssignVariableOp_6¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9ç
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:1*
dtype0*
valueB1B5layer_with_weights-0/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-0/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-0/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-2/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-2/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-2/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-4/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-4/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-6/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-6/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-6/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-8/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-8/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-8/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-10/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-10/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-10/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-12/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-12/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-12/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-14/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-14/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-14/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-14/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-15/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-15/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHÒ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:1*
dtype0*u
valuelBj1B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*Ú
_output_shapesÇ
Ä:::::::::::::::::::::::::::::::::::::::::::::::::*?
dtypes5
321[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOpAssignVariableOp-assignvariableop_batch_normalization_32_gammaIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_1AssignVariableOp.assignvariableop_1_batch_normalization_32_betaIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:¤
AssignVariableOp_2AssignVariableOp5assignvariableop_2_batch_normalization_32_moving_meanIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_3AssignVariableOp9assignvariableop_3_batch_normalization_32_moving_varianceIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_4AssignVariableOp#assignvariableop_4_conv2d_20_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_5AssignVariableOp!assignvariableop_5_conv2d_20_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_6AssignVariableOp/assignvariableop_6_batch_normalization_33_gammaIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_7AssignVariableOp.assignvariableop_7_batch_normalization_33_betaIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:¤
AssignVariableOp_8AssignVariableOp5assignvariableop_8_batch_normalization_33_moving_meanIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_9AssignVariableOp9assignvariableop_9_batch_normalization_33_moving_varianceIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_10AssignVariableOp$assignvariableop_10_conv2d_21_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_11AssignVariableOp"assignvariableop_11_conv2d_21_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:¡
AssignVariableOp_12AssignVariableOp0assignvariableop_12_batch_normalization_34_gammaIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
: 
AssignVariableOp_13AssignVariableOp/assignvariableop_13_batch_normalization_34_betaIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:§
AssignVariableOp_14AssignVariableOp6assignvariableop_14_batch_normalization_34_moving_meanIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:«
AssignVariableOp_15AssignVariableOp:assignvariableop_15_batch_normalization_34_moving_varianceIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_16AssignVariableOp$assignvariableop_16_conv2d_22_kernelIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_17AssignVariableOp"assignvariableop_17_conv2d_22_biasIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:¡
AssignVariableOp_18AssignVariableOp0assignvariableop_18_batch_normalization_35_gammaIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
: 
AssignVariableOp_19AssignVariableOp/assignvariableop_19_batch_normalization_35_betaIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:§
AssignVariableOp_20AssignVariableOp6assignvariableop_20_batch_normalization_35_moving_meanIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:«
AssignVariableOp_21AssignVariableOp:assignvariableop_21_batch_normalization_35_moving_varianceIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_22AssignVariableOp$assignvariableop_22_conv2d_23_kernelIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_23AssignVariableOp"assignvariableop_23_conv2d_23_biasIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:¡
AssignVariableOp_24AssignVariableOp0assignvariableop_24_batch_normalization_36_gammaIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
: 
AssignVariableOp_25AssignVariableOp/assignvariableop_25_batch_normalization_36_betaIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:§
AssignVariableOp_26AssignVariableOp6assignvariableop_26_batch_normalization_36_moving_meanIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:«
AssignVariableOp_27AssignVariableOp:assignvariableop_27_batch_normalization_36_moving_varianceIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_28AssignVariableOp$assignvariableop_28_conv2d_24_kernelIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_29AssignVariableOp"assignvariableop_29_conv2d_24_biasIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:¡
AssignVariableOp_30AssignVariableOp0assignvariableop_30_batch_normalization_37_gammaIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
: 
AssignVariableOp_31AssignVariableOp/assignvariableop_31_batch_normalization_37_betaIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:§
AssignVariableOp_32AssignVariableOp6assignvariableop_32_batch_normalization_37_moving_meanIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:«
AssignVariableOp_33AssignVariableOp:assignvariableop_33_batch_normalization_37_moving_varianceIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_34AssignVariableOp&assignvariableop_34_features512_kernelIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_35AssignVariableOp$assignvariableop_35_features512_biasIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:¡
AssignVariableOp_36AssignVariableOp0assignvariableop_36_batch_normalization_38_gammaIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
: 
AssignVariableOp_37AssignVariableOp/assignvariableop_37_batch_normalization_38_betaIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:§
AssignVariableOp_38AssignVariableOp6assignvariableop_38_batch_normalization_38_moving_meanIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:«
AssignVariableOp_39AssignVariableOp:assignvariableop_39_batch_normalization_38_moving_varianceIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_40AssignVariableOp&assignvariableop_40_features256_kernelIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_41AssignVariableOp$assignvariableop_41_features256_biasIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:¡
AssignVariableOp_42AssignVariableOp0assignvariableop_42_batch_normalization_39_gammaIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
: 
AssignVariableOp_43AssignVariableOp/assignvariableop_43_batch_normalization_39_betaIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:§
AssignVariableOp_44AssignVariableOp6assignvariableop_44_batch_normalization_39_moving_meanIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:«
AssignVariableOp_45AssignVariableOp:assignvariableop_45_batch_normalization_39_moving_varianceIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_46AssignVariableOp"assignvariableop_46_dense_4_kernelIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_47AssignVariableOp assignvariableop_47_dense_4_biasIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ï
Identity_48Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_49IdentityIdentity_48:output:0^NoOp_1*
T0*
_output_shapes
: Ü
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_49Identity_49:output:0*u
_input_shapesd
b: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
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
AssignVariableOp_47AssignVariableOp_472(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
Ì

Q__inference_batch_normalization_35_layer_call_and_return_conditional_losses_15423

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0È
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : : :*
epsilon%o:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ °
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
»
L
0__inference_max_pooling2d_21_layer_call_fn_18324

inputs
identityÜ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_max_pooling2d_21_layer_call_and_return_conditional_losses_15398
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
»
L
0__inference_max_pooling2d_23_layer_call_fn_18580

inputs
identityÜ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_max_pooling2d_23_layer_call_and_return_conditional_losses_15550
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

g
K__inference_max_pooling2d_23_layer_call_and_return_conditional_losses_18585

inputs
identity¢
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

c
*__inference_dropout_28_layer_call_fn_18211

inputs
identity¢StatefulPartitionedCallË
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_28_layer_call_and_return_conditional_losses_16498w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ122
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
 
_user_specified_nameinputs
Ì

Q__inference_batch_normalization_36_layer_call_and_return_conditional_losses_15499

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0È
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@°
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
ø
c
E__inference_dropout_29_layer_call_and_return_conditional_losses_18344

inputs

identity_1V
IdentityIdentityinputs*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
c

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ
:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs

À
Q__inference_batch_normalization_36_layer_call_and_return_conditional_losses_15530

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity¢AssignNewValue¢AssignNewValue_1¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0Ö
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
exponential_avg_factor%
×#<Æ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(Ð
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@Ô
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
	
Ñ
6__inference_batch_normalization_36_layer_call_fn_18529

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_batch_normalization_36_layer_call_and_return_conditional_losses_15530
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs

c
*__inference_dropout_31_layer_call_fn_18595

inputs
identity¢StatefulPartitionedCallË
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_31_layer_call_and_return_conditional_losses_16381w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
ø
c
E__inference_dropout_31_layer_call_and_return_conditional_losses_18600

inputs

identity_1V
IdentityIdentityinputs*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@c

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs

g
K__inference_max_pooling2d_20_layer_call_and_return_conditional_losses_15322

inputs
identity¢
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ç
_
C__inference_re_lu_31_layer_call_and_return_conditional_losses_15975

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@b
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
ø
c
E__inference_dropout_29_layer_call_and_return_conditional_losses_15903

inputs

identity_1V
IdentityIdentityinputs*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
c

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ
:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs
	
Ñ
6__inference_batch_normalization_32_layer_call_fn_18051

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_batch_normalization_32_layer_call_and_return_conditional_losses_15207
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ì

Q__inference_batch_normalization_34_layer_call_and_return_conditional_losses_18291

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0È
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ°
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

g
K__inference_max_pooling2d_22_layer_call_and_return_conditional_losses_15474

inputs
identity¢
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¾
D
(__inference_re_lu_30_layer_call_fn_18442

inputs
identity¹
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_re_lu_30_layer_call_and_return_conditional_losses_15935h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ
 :W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
 
_user_specified_nameinputs
´
E
)__inference_flatten_4_layer_call_fn_18745

inputs
identity³
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_flatten_4_layer_call_and_return_conditional_losses_16031a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ë
_
C__inference_re_lu_34_layer_call_and_return_conditional_losses_16102

inputs
identityG
ReluReluinputs*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ç
_
C__inference_re_lu_28_layer_call_and_return_conditional_losses_18191

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿc(b
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿc("
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿc(:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿc(
 
_user_specified_nameinputs
®

ÿ
D__inference_conv2d_24_layer_call_and_return_conditional_losses_18631

inputs9
conv2d_readvariableop_resource:@.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
IdentityIdentityBiasAdd:output:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
¾
D
(__inference_re_lu_29_layer_call_fn_18314

inputs
identity¹
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_re_lu_29_layer_call_and_return_conditional_losses_15895h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ1:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
 
_user_specified_nameinputs

Ä
Q__inference_batch_normalization_37_layer_call_and_return_conditional_losses_18693

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identity¢AssignNewValue¢AssignNewValue_1¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0Û
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
exponential_avg_factor%
×#<Æ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(Ð
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÔ
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
µ 
á
G__inference_sequential_4_layer_call_and_return_conditional_losses_16129

inputs*
batch_normalization_32_15816:*
batch_normalization_32_15818:*
batch_normalization_32_15820:*
batch_normalization_32_15822:)
conv2d_20_15836:
conv2d_20_15838:*
batch_normalization_33_15841:*
batch_normalization_33_15843:*
batch_normalization_33_15845:*
batch_normalization_33_15847:)
conv2d_21_15876:
conv2d_21_15878:*
batch_normalization_34_15881:*
batch_normalization_34_15883:*
batch_normalization_34_15885:*
batch_normalization_34_15887:)
conv2d_22_15916: 
conv2d_22_15918: *
batch_normalization_35_15921: *
batch_normalization_35_15923: *
batch_normalization_35_15925: *
batch_normalization_35_15927: )
conv2d_23_15956: @
conv2d_23_15958:@*
batch_normalization_36_15961:@*
batch_normalization_36_15963:@*
batch_normalization_36_15965:@*
batch_normalization_36_15967:@*
conv2d_24_15996:@
conv2d_24_15998:	+
batch_normalization_37_16001:	+
batch_normalization_37_16003:	+
batch_normalization_37_16005:	+
batch_normalization_37_16007:	%
features512_16044:
 
features512_16046:	+
batch_normalization_38_16049:	+
batch_normalization_38_16051:	+
batch_normalization_38_16053:	+
batch_normalization_38_16055:	%
features256_16083:
 
features256_16085:	+
batch_normalization_39_16088:	+
batch_normalization_39_16090:	+
batch_normalization_39_16092:	+
batch_normalization_39_16094:	 
dense_4_16123:	
dense_4_16125:
identity¢.batch_normalization_32/StatefulPartitionedCall¢.batch_normalization_33/StatefulPartitionedCall¢.batch_normalization_34/StatefulPartitionedCall¢.batch_normalization_35/StatefulPartitionedCall¢.batch_normalization_36/StatefulPartitionedCall¢.batch_normalization_37/StatefulPartitionedCall¢.batch_normalization_38/StatefulPartitionedCall¢.batch_normalization_39/StatefulPartitionedCall¢!conv2d_20/StatefulPartitionedCall¢!conv2d_21/StatefulPartitionedCall¢!conv2d_22/StatefulPartitionedCall¢!conv2d_23/StatefulPartitionedCall¢!conv2d_24/StatefulPartitionedCall¢dense_4/StatefulPartitionedCall¢#features256/StatefulPartitionedCall¢#features512/StatefulPartitionedCallÄ
reshape_4/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿc(* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_reshape_4_layer_call_and_return_conditional_losses_15814
.batch_normalization_32/StatefulPartitionedCallStatefulPartitionedCall"reshape_4/PartitionedCall:output:0batch_normalization_32_15816batch_normalization_32_15818batch_normalization_32_15820batch_normalization_32_15822*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿc(*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_batch_normalization_32_layer_call_and_return_conditional_losses_15207­
!conv2d_20/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_32/StatefulPartitionedCall:output:0conv2d_20_15836conv2d_20_15838*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿc(*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_20_layer_call_and_return_conditional_losses_15835
.batch_normalization_33/StatefulPartitionedCallStatefulPartitionedCall*conv2d_20/StatefulPartitionedCall:output:0batch_normalization_33_15841batch_normalization_33_15843batch_normalization_33_15845batch_normalization_33_15847*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿc(*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_batch_normalization_33_layer_call_and_return_conditional_losses_15271ó
re_lu_28/PartitionedCallPartitionedCall7batch_normalization_33/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿc(* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_re_lu_28_layer_call_and_return_conditional_losses_15855í
 max_pooling2d_20/PartitionedCallPartitionedCall!re_lu_28/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_max_pooling2d_20_layer_call_and_return_conditional_losses_15322é
dropout_28/PartitionedCallPartitionedCall)max_pooling2d_20/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_28_layer_call_and_return_conditional_losses_15863
!conv2d_21/StatefulPartitionedCallStatefulPartitionedCall#dropout_28/PartitionedCall:output:0conv2d_21_15876conv2d_21_15878*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_21_layer_call_and_return_conditional_losses_15875
.batch_normalization_34/StatefulPartitionedCallStatefulPartitionedCall*conv2d_21/StatefulPartitionedCall:output:0batch_normalization_34_15881batch_normalization_34_15883batch_normalization_34_15885batch_normalization_34_15887*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_batch_normalization_34_layer_call_and_return_conditional_losses_15347ó
re_lu_29/PartitionedCallPartitionedCall7batch_normalization_34/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_re_lu_29_layer_call_and_return_conditional_losses_15895í
 max_pooling2d_21/PartitionedCallPartitionedCall!re_lu_29/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_max_pooling2d_21_layer_call_and_return_conditional_losses_15398é
dropout_29/PartitionedCallPartitionedCall)max_pooling2d_21/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_29_layer_call_and_return_conditional_losses_15903
!conv2d_22/StatefulPartitionedCallStatefulPartitionedCall#dropout_29/PartitionedCall:output:0conv2d_22_15916conv2d_22_15918*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_22_layer_call_and_return_conditional_losses_15915
.batch_normalization_35/StatefulPartitionedCallStatefulPartitionedCall*conv2d_22/StatefulPartitionedCall:output:0batch_normalization_35_15921batch_normalization_35_15923batch_normalization_35_15925batch_normalization_35_15927*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_batch_normalization_35_layer_call_and_return_conditional_losses_15423ó
re_lu_30/PartitionedCallPartitionedCall7batch_normalization_35/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_re_lu_30_layer_call_and_return_conditional_losses_15935í
 max_pooling2d_22/PartitionedCallPartitionedCall!re_lu_30/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_max_pooling2d_22_layer_call_and_return_conditional_losses_15474é
dropout_30/PartitionedCallPartitionedCall)max_pooling2d_22/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_30_layer_call_and_return_conditional_losses_15943
!conv2d_23/StatefulPartitionedCallStatefulPartitionedCall#dropout_30/PartitionedCall:output:0conv2d_23_15956conv2d_23_15958*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_23_layer_call_and_return_conditional_losses_15955
.batch_normalization_36/StatefulPartitionedCallStatefulPartitionedCall*conv2d_23/StatefulPartitionedCall:output:0batch_normalization_36_15961batch_normalization_36_15963batch_normalization_36_15965batch_normalization_36_15967*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_batch_normalization_36_layer_call_and_return_conditional_losses_15499ó
re_lu_31/PartitionedCallPartitionedCall7batch_normalization_36/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_re_lu_31_layer_call_and_return_conditional_losses_15975í
 max_pooling2d_23/PartitionedCallPartitionedCall!re_lu_31/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_max_pooling2d_23_layer_call_and_return_conditional_losses_15550é
dropout_31/PartitionedCallPartitionedCall)max_pooling2d_23/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_31_layer_call_and_return_conditional_losses_15983
!conv2d_24/StatefulPartitionedCallStatefulPartitionedCall#dropout_31/PartitionedCall:output:0conv2d_24_15996conv2d_24_15998*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_24_layer_call_and_return_conditional_losses_15995
.batch_normalization_37/StatefulPartitionedCallStatefulPartitionedCall*conv2d_24/StatefulPartitionedCall:output:0batch_normalization_37_16001batch_normalization_37_16003batch_normalization_37_16005batch_normalization_37_16007*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_batch_normalization_37_layer_call_and_return_conditional_losses_15575ô
re_lu_32/PartitionedCallPartitionedCall7batch_normalization_37/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_re_lu_32_layer_call_and_return_conditional_losses_16015î
 max_pooling2d_24/PartitionedCallPartitionedCall!re_lu_32/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_max_pooling2d_24_layer_call_and_return_conditional_losses_15626ê
dropout_32/PartitionedCallPartitionedCall)max_pooling2d_24/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_32_layer_call_and_return_conditional_losses_16023Ú
flatten_4/PartitionedCallPartitionedCall#dropout_32/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_flatten_4_layer_call_and_return_conditional_losses_16031
#features512/StatefulPartitionedCallStatefulPartitionedCall"flatten_4/PartitionedCall:output:0features512_16044features512_16046*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_features512_layer_call_and_return_conditional_losses_16043
.batch_normalization_38/StatefulPartitionedCallStatefulPartitionedCall,features512/StatefulPartitionedCall:output:0batch_normalization_38_16049batch_normalization_38_16051batch_normalization_38_16053batch_normalization_38_16055*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_batch_normalization_38_layer_call_and_return_conditional_losses_15653ì
re_lu_33/PartitionedCallPartitionedCall7batch_normalization_38/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_re_lu_33_layer_call_and_return_conditional_losses_16063Ú
dropout_33/PartitionedCallPartitionedCall!re_lu_33/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_33_layer_call_and_return_conditional_losses_16070
#features256/StatefulPartitionedCallStatefulPartitionedCall#dropout_33/PartitionedCall:output:0features256_16083features256_16085*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_features256_layer_call_and_return_conditional_losses_16082
.batch_normalization_39/StatefulPartitionedCallStatefulPartitionedCall,features256/StatefulPartitionedCall:output:0batch_normalization_39_16088batch_normalization_39_16090batch_normalization_39_16092batch_normalization_39_16094*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_batch_normalization_39_layer_call_and_return_conditional_losses_15735ì
re_lu_34/PartitionedCallPartitionedCall7batch_normalization_39/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_re_lu_34_layer_call_and_return_conditional_losses_16102Ú
dropout_34/PartitionedCallPartitionedCall!re_lu_34/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_34_layer_call_and_return_conditional_losses_16109
dense_4/StatefulPartitionedCallStatefulPartitionedCall#dropout_34/PartitionedCall:output:0dense_4_16123dense_4_16125*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_4_layer_call_and_return_conditional_losses_16122w
IdentityIdentity(dense_4/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿð
NoOpNoOp/^batch_normalization_32/StatefulPartitionedCall/^batch_normalization_33/StatefulPartitionedCall/^batch_normalization_34/StatefulPartitionedCall/^batch_normalization_35/StatefulPartitionedCall/^batch_normalization_36/StatefulPartitionedCall/^batch_normalization_37/StatefulPartitionedCall/^batch_normalization_38/StatefulPartitionedCall/^batch_normalization_39/StatefulPartitionedCall"^conv2d_20/StatefulPartitionedCall"^conv2d_21/StatefulPartitionedCall"^conv2d_22/StatefulPartitionedCall"^conv2d_23/StatefulPartitionedCall"^conv2d_24/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall$^features256/StatefulPartitionedCall$^features512/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapesy
w:ÿÿÿÿÿÿÿÿÿc(: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2`
.batch_normalization_32/StatefulPartitionedCall.batch_normalization_32/StatefulPartitionedCall2`
.batch_normalization_33/StatefulPartitionedCall.batch_normalization_33/StatefulPartitionedCall2`
.batch_normalization_34/StatefulPartitionedCall.batch_normalization_34/StatefulPartitionedCall2`
.batch_normalization_35/StatefulPartitionedCall.batch_normalization_35/StatefulPartitionedCall2`
.batch_normalization_36/StatefulPartitionedCall.batch_normalization_36/StatefulPartitionedCall2`
.batch_normalization_37/StatefulPartitionedCall.batch_normalization_37/StatefulPartitionedCall2`
.batch_normalization_38/StatefulPartitionedCall.batch_normalization_38/StatefulPartitionedCall2`
.batch_normalization_39/StatefulPartitionedCall.batch_normalization_39/StatefulPartitionedCall2F
!conv2d_20/StatefulPartitionedCall!conv2d_20/StatefulPartitionedCall2F
!conv2d_21/StatefulPartitionedCall!conv2d_21/StatefulPartitionedCall2F
!conv2d_22/StatefulPartitionedCall!conv2d_22/StatefulPartitionedCall2F
!conv2d_23/StatefulPartitionedCall!conv2d_23/StatefulPartitionedCall2F
!conv2d_24/StatefulPartitionedCall!conv2d_24/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2J
#features256/StatefulPartitionedCall#features256/StatefulPartitionedCall2J
#features512/StatefulPartitionedCall#features512/StatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿc(
 
_user_specified_nameinputs
Â
D
(__inference_re_lu_32_layer_call_fn_18698

inputs
identityº
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_re_lu_32_layer_call_and_return_conditional_losses_16015i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

g
K__inference_max_pooling2d_21_layer_call_and_return_conditional_losses_18329

inputs
identity¢
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

c
*__inference_dropout_29_layer_call_fn_18339

inputs
identity¢StatefulPartitionedCallË
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_29_layer_call_and_return_conditional_losses_16459w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ
22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs
»
L
0__inference_max_pooling2d_24_layer_call_fn_18708

inputs
identityÜ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_max_pooling2d_24_layer_call_and_return_conditional_losses_15626
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
³

d
E__inference_dropout_29_layer_call_and_return_conditional_losses_18356

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *«ªª?l
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  >®
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
w
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
q
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
a
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ
:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs
§

ý
D__inference_conv2d_21_layer_call_and_return_conditional_losses_18247

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ1: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
 
_user_specified_nameinputs
Ì

Q__inference_batch_normalization_32_layer_call_and_return_conditional_losses_18082

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0È
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ°
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
	
Õ
6__inference_batch_normalization_37_layer_call_fn_18644

inputs
unknown:	
	unknown_0:	
	unknown_1:	
	unknown_2:	
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_batch_normalization_37_layer_call_and_return_conditional_losses_15575
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ô	
ú
F__inference_features512_layer_call_and_return_conditional_losses_18770

inputs2
matmul_readvariableop_resource:
.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
IdentityIdentityBiasAdd:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

ÿ0
G__inference_sequential_4_layer_call_and_return_conditional_losses_18019

inputs<
.batch_normalization_32_readvariableop_resource:>
0batch_normalization_32_readvariableop_1_resource:M
?batch_normalization_32_fusedbatchnormv3_readvariableop_resource:O
Abatch_normalization_32_fusedbatchnormv3_readvariableop_1_resource:B
(conv2d_20_conv2d_readvariableop_resource:7
)conv2d_20_biasadd_readvariableop_resource:<
.batch_normalization_33_readvariableop_resource:>
0batch_normalization_33_readvariableop_1_resource:M
?batch_normalization_33_fusedbatchnormv3_readvariableop_resource:O
Abatch_normalization_33_fusedbatchnormv3_readvariableop_1_resource:B
(conv2d_21_conv2d_readvariableop_resource:7
)conv2d_21_biasadd_readvariableop_resource:<
.batch_normalization_34_readvariableop_resource:>
0batch_normalization_34_readvariableop_1_resource:M
?batch_normalization_34_fusedbatchnormv3_readvariableop_resource:O
Abatch_normalization_34_fusedbatchnormv3_readvariableop_1_resource:B
(conv2d_22_conv2d_readvariableop_resource: 7
)conv2d_22_biasadd_readvariableop_resource: <
.batch_normalization_35_readvariableop_resource: >
0batch_normalization_35_readvariableop_1_resource: M
?batch_normalization_35_fusedbatchnormv3_readvariableop_resource: O
Abatch_normalization_35_fusedbatchnormv3_readvariableop_1_resource: B
(conv2d_23_conv2d_readvariableop_resource: @7
)conv2d_23_biasadd_readvariableop_resource:@<
.batch_normalization_36_readvariableop_resource:@>
0batch_normalization_36_readvariableop_1_resource:@M
?batch_normalization_36_fusedbatchnormv3_readvariableop_resource:@O
Abatch_normalization_36_fusedbatchnormv3_readvariableop_1_resource:@C
(conv2d_24_conv2d_readvariableop_resource:@8
)conv2d_24_biasadd_readvariableop_resource:	=
.batch_normalization_37_readvariableop_resource:	?
0batch_normalization_37_readvariableop_1_resource:	N
?batch_normalization_37_fusedbatchnormv3_readvariableop_resource:	P
Abatch_normalization_37_fusedbatchnormv3_readvariableop_1_resource:	>
*features512_matmul_readvariableop_resource:
:
+features512_biasadd_readvariableop_resource:	M
>batch_normalization_38_assignmovingavg_readvariableop_resource:	O
@batch_normalization_38_assignmovingavg_1_readvariableop_resource:	K
<batch_normalization_38_batchnorm_mul_readvariableop_resource:	G
8batch_normalization_38_batchnorm_readvariableop_resource:	>
*features256_matmul_readvariableop_resource:
:
+features256_biasadd_readvariableop_resource:	M
>batch_normalization_39_assignmovingavg_readvariableop_resource:	O
@batch_normalization_39_assignmovingavg_1_readvariableop_resource:	K
<batch_normalization_39_batchnorm_mul_readvariableop_resource:	G
8batch_normalization_39_batchnorm_readvariableop_resource:	9
&dense_4_matmul_readvariableop_resource:	5
'dense_4_biasadd_readvariableop_resource:
identity¢%batch_normalization_32/AssignNewValue¢'batch_normalization_32/AssignNewValue_1¢6batch_normalization_32/FusedBatchNormV3/ReadVariableOp¢8batch_normalization_32/FusedBatchNormV3/ReadVariableOp_1¢%batch_normalization_32/ReadVariableOp¢'batch_normalization_32/ReadVariableOp_1¢%batch_normalization_33/AssignNewValue¢'batch_normalization_33/AssignNewValue_1¢6batch_normalization_33/FusedBatchNormV3/ReadVariableOp¢8batch_normalization_33/FusedBatchNormV3/ReadVariableOp_1¢%batch_normalization_33/ReadVariableOp¢'batch_normalization_33/ReadVariableOp_1¢%batch_normalization_34/AssignNewValue¢'batch_normalization_34/AssignNewValue_1¢6batch_normalization_34/FusedBatchNormV3/ReadVariableOp¢8batch_normalization_34/FusedBatchNormV3/ReadVariableOp_1¢%batch_normalization_34/ReadVariableOp¢'batch_normalization_34/ReadVariableOp_1¢%batch_normalization_35/AssignNewValue¢'batch_normalization_35/AssignNewValue_1¢6batch_normalization_35/FusedBatchNormV3/ReadVariableOp¢8batch_normalization_35/FusedBatchNormV3/ReadVariableOp_1¢%batch_normalization_35/ReadVariableOp¢'batch_normalization_35/ReadVariableOp_1¢%batch_normalization_36/AssignNewValue¢'batch_normalization_36/AssignNewValue_1¢6batch_normalization_36/FusedBatchNormV3/ReadVariableOp¢8batch_normalization_36/FusedBatchNormV3/ReadVariableOp_1¢%batch_normalization_36/ReadVariableOp¢'batch_normalization_36/ReadVariableOp_1¢%batch_normalization_37/AssignNewValue¢'batch_normalization_37/AssignNewValue_1¢6batch_normalization_37/FusedBatchNormV3/ReadVariableOp¢8batch_normalization_37/FusedBatchNormV3/ReadVariableOp_1¢%batch_normalization_37/ReadVariableOp¢'batch_normalization_37/ReadVariableOp_1¢&batch_normalization_38/AssignMovingAvg¢5batch_normalization_38/AssignMovingAvg/ReadVariableOp¢(batch_normalization_38/AssignMovingAvg_1¢7batch_normalization_38/AssignMovingAvg_1/ReadVariableOp¢/batch_normalization_38/batchnorm/ReadVariableOp¢3batch_normalization_38/batchnorm/mul/ReadVariableOp¢&batch_normalization_39/AssignMovingAvg¢5batch_normalization_39/AssignMovingAvg/ReadVariableOp¢(batch_normalization_39/AssignMovingAvg_1¢7batch_normalization_39/AssignMovingAvg_1/ReadVariableOp¢/batch_normalization_39/batchnorm/ReadVariableOp¢3batch_normalization_39/batchnorm/mul/ReadVariableOp¢ conv2d_20/BiasAdd/ReadVariableOp¢conv2d_20/Conv2D/ReadVariableOp¢ conv2d_21/BiasAdd/ReadVariableOp¢conv2d_21/Conv2D/ReadVariableOp¢ conv2d_22/BiasAdd/ReadVariableOp¢conv2d_22/Conv2D/ReadVariableOp¢ conv2d_23/BiasAdd/ReadVariableOp¢conv2d_23/Conv2D/ReadVariableOp¢ conv2d_24/BiasAdd/ReadVariableOp¢conv2d_24/Conv2D/ReadVariableOp¢dense_4/BiasAdd/ReadVariableOp¢dense_4/MatMul/ReadVariableOp¢"features256/BiasAdd/ReadVariableOp¢!features256/MatMul/ReadVariableOp¢"features512/BiasAdd/ReadVariableOp¢!features512/MatMul/ReadVariableOpE
reshape_4/ShapeShapeinputs*
T0*
_output_shapes
:g
reshape_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: i
reshape_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
reshape_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
reshape_4/strided_sliceStridedSlicereshape_4/Shape:output:0&reshape_4/strided_slice/stack:output:0(reshape_4/strided_slice/stack_1:output:0(reshape_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask[
reshape_4/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :c[
reshape_4/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :([
reshape_4/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :Û
reshape_4/Reshape/shapePack reshape_4/strided_slice:output:0"reshape_4/Reshape/shape/1:output:0"reshape_4/Reshape/shape/2:output:0"reshape_4/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:
reshape_4/ReshapeReshapeinputs reshape_4/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿc(
%batch_normalization_32/ReadVariableOpReadVariableOp.batch_normalization_32_readvariableop_resource*
_output_shapes
:*
dtype0
'batch_normalization_32/ReadVariableOp_1ReadVariableOp0batch_normalization_32_readvariableop_1_resource*
_output_shapes
:*
dtype0²
6batch_normalization_32/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_32_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0¶
8batch_normalization_32/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_32_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0Ë
'batch_normalization_32/FusedBatchNormV3FusedBatchNormV3reshape_4/Reshape:output:0-batch_normalization_32/ReadVariableOp:value:0/batch_normalization_32/ReadVariableOp_1:value:0>batch_normalization_32/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_32/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿc(:::::*
epsilon%o:*
exponential_avg_factor%
×#<¢
%batch_normalization_32/AssignNewValueAssignVariableOp?batch_normalization_32_fusedbatchnormv3_readvariableop_resource4batch_normalization_32/FusedBatchNormV3:batch_mean:07^batch_normalization_32/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(¬
'batch_normalization_32/AssignNewValue_1AssignVariableOpAbatch_normalization_32_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_32/FusedBatchNormV3:batch_variance:09^batch_normalization_32/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(
conv2d_20/Conv2D/ReadVariableOpReadVariableOp(conv2d_20_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0Ò
conv2d_20/Conv2DConv2D+batch_normalization_32/FusedBatchNormV3:y:0'conv2d_20/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿc(*
paddingSAME*
strides

 conv2d_20/BiasAdd/ReadVariableOpReadVariableOp)conv2d_20_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
conv2d_20/BiasAddBiasAddconv2d_20/Conv2D:output:0(conv2d_20/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿc(
%batch_normalization_33/ReadVariableOpReadVariableOp.batch_normalization_33_readvariableop_resource*
_output_shapes
:*
dtype0
'batch_normalization_33/ReadVariableOp_1ReadVariableOp0batch_normalization_33_readvariableop_1_resource*
_output_shapes
:*
dtype0²
6batch_normalization_33/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_33_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0¶
8batch_normalization_33/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_33_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0Ë
'batch_normalization_33/FusedBatchNormV3FusedBatchNormV3conv2d_20/BiasAdd:output:0-batch_normalization_33/ReadVariableOp:value:0/batch_normalization_33/ReadVariableOp_1:value:0>batch_normalization_33/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_33/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿc(:::::*
epsilon%o:*
exponential_avg_factor%
×#<¢
%batch_normalization_33/AssignNewValueAssignVariableOp?batch_normalization_33_fusedbatchnormv3_readvariableop_resource4batch_normalization_33/FusedBatchNormV3:batch_mean:07^batch_normalization_33/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(¬
'batch_normalization_33/AssignNewValue_1AssignVariableOpAbatch_normalization_33_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_33/FusedBatchNormV3:batch_variance:09^batch_normalization_33/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(|
re_lu_28/ReluRelu+batch_normalization_33/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿc(­
max_pooling2d_20/MaxPoolMaxPoolre_lu_28/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*
ksize
*
paddingVALID*
strides
]
dropout_28/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *«ªª?
dropout_28/dropout/MulMul!max_pooling2d_20/MaxPool:output:0!dropout_28/dropout/Const:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1i
dropout_28/dropout/ShapeShape!max_pooling2d_20/MaxPool:output:0*
T0*
_output_shapes
:ª
/dropout_28/dropout/random_uniform/RandomUniformRandomUniform!dropout_28/dropout/Shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*
dtype0f
!dropout_28/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  >Ï
dropout_28/dropout/GreaterEqualGreaterEqual8dropout_28/dropout/random_uniform/RandomUniform:output:0*dropout_28/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
dropout_28/dropout/CastCast#dropout_28/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
dropout_28/dropout/Mul_1Muldropout_28/dropout/Mul:z:0dropout_28/dropout/Cast:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
conv2d_21/Conv2D/ReadVariableOpReadVariableOp(conv2d_21_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0Ã
conv2d_21/Conv2DConv2Ddropout_28/dropout/Mul_1:z:0'conv2d_21/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*
paddingSAME*
strides

 conv2d_21/BiasAdd/ReadVariableOpReadVariableOp)conv2d_21_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
conv2d_21/BiasAddBiasAddconv2d_21/Conv2D:output:0(conv2d_21/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
%batch_normalization_34/ReadVariableOpReadVariableOp.batch_normalization_34_readvariableop_resource*
_output_shapes
:*
dtype0
'batch_normalization_34/ReadVariableOp_1ReadVariableOp0batch_normalization_34_readvariableop_1_resource*
_output_shapes
:*
dtype0²
6batch_normalization_34/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_34_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0¶
8batch_normalization_34/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_34_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0Ë
'batch_normalization_34/FusedBatchNormV3FusedBatchNormV3conv2d_21/BiasAdd:output:0-batch_normalization_34/ReadVariableOp:value:0/batch_normalization_34/ReadVariableOp_1:value:0>batch_normalization_34/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_34/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ1:::::*
epsilon%o:*
exponential_avg_factor%
×#<¢
%batch_normalization_34/AssignNewValueAssignVariableOp?batch_normalization_34_fusedbatchnormv3_readvariableop_resource4batch_normalization_34/FusedBatchNormV3:batch_mean:07^batch_normalization_34/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(¬
'batch_normalization_34/AssignNewValue_1AssignVariableOpAbatch_normalization_34_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_34/FusedBatchNormV3:batch_variance:09^batch_normalization_34/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(|
re_lu_29/ReluRelu+batch_normalization_34/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1­
max_pooling2d_21/MaxPoolMaxPoolre_lu_29/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
ksize
*
paddingVALID*
strides
]
dropout_29/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *«ªª?
dropout_29/dropout/MulMul!max_pooling2d_21/MaxPool:output:0!dropout_29/dropout/Const:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
i
dropout_29/dropout/ShapeShape!max_pooling2d_21/MaxPool:output:0*
T0*
_output_shapes
:ª
/dropout_29/dropout/random_uniform/RandomUniformRandomUniform!dropout_29/dropout/Shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
dtype0f
!dropout_29/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  >Ï
dropout_29/dropout/GreaterEqualGreaterEqual8dropout_29/dropout/random_uniform/RandomUniform:output:0*dropout_29/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

dropout_29/dropout/CastCast#dropout_29/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

dropout_29/dropout/Mul_1Muldropout_29/dropout/Mul:z:0dropout_29/dropout/Cast:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

conv2d_22/Conv2D/ReadVariableOpReadVariableOp(conv2d_22_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0Ã
conv2d_22/Conv2DConv2Ddropout_29/dropout/Mul_1:z:0'conv2d_22/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 *
paddingSAME*
strides

 conv2d_22/BiasAdd/ReadVariableOpReadVariableOp)conv2d_22_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0
conv2d_22/BiasAddBiasAddconv2d_22/Conv2D:output:0(conv2d_22/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
%batch_normalization_35/ReadVariableOpReadVariableOp.batch_normalization_35_readvariableop_resource*
_output_shapes
: *
dtype0
'batch_normalization_35/ReadVariableOp_1ReadVariableOp0batch_normalization_35_readvariableop_1_resource*
_output_shapes
: *
dtype0²
6batch_normalization_35/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_35_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0¶
8batch_normalization_35/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_35_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0Ë
'batch_normalization_35/FusedBatchNormV3FusedBatchNormV3conv2d_22/BiasAdd:output:0-batch_normalization_35/ReadVariableOp:value:0/batch_normalization_35/ReadVariableOp_1:value:0>batch_normalization_35/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_35/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ
 : : : : :*
epsilon%o:*
exponential_avg_factor%
×#<¢
%batch_normalization_35/AssignNewValueAssignVariableOp?batch_normalization_35_fusedbatchnormv3_readvariableop_resource4batch_normalization_35/FusedBatchNormV3:batch_mean:07^batch_normalization_35/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(¬
'batch_normalization_35/AssignNewValue_1AssignVariableOpAbatch_normalization_35_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_35/FusedBatchNormV3:batch_variance:09^batch_normalization_35/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(|
re_lu_30/ReluRelu+batch_normalization_35/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 ­
max_pooling2d_22/MaxPoolMaxPoolre_lu_30/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
ksize
*
paddingVALID*
strides
]
dropout_30/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *«ªª?
dropout_30/dropout/MulMul!max_pooling2d_22/MaxPool:output:0!dropout_30/dropout/Const:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ i
dropout_30/dropout/ShapeShape!max_pooling2d_22/MaxPool:output:0*
T0*
_output_shapes
:ª
/dropout_30/dropout/random_uniform/RandomUniformRandomUniform!dropout_30/dropout/Shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
dtype0f
!dropout_30/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  >Ï
dropout_30/dropout/GreaterEqualGreaterEqual8dropout_30/dropout/random_uniform/RandomUniform:output:0*dropout_30/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
dropout_30/dropout/CastCast#dropout_30/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
dropout_30/dropout/Mul_1Muldropout_30/dropout/Mul:z:0dropout_30/dropout/Cast:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
conv2d_23/Conv2D/ReadVariableOpReadVariableOp(conv2d_23_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0Ã
conv2d_23/Conv2DConv2Ddropout_30/dropout/Mul_1:z:0'conv2d_23/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingSAME*
strides

 conv2d_23/BiasAdd/ReadVariableOpReadVariableOp)conv2d_23_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
conv2d_23/BiasAddBiasAddconv2d_23/Conv2D:output:0(conv2d_23/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
%batch_normalization_36/ReadVariableOpReadVariableOp.batch_normalization_36_readvariableop_resource*
_output_shapes
:@*
dtype0
'batch_normalization_36/ReadVariableOp_1ReadVariableOp0batch_normalization_36_readvariableop_1_resource*
_output_shapes
:@*
dtype0²
6batch_normalization_36/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_36_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0¶
8batch_normalization_36/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_36_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0Ë
'batch_normalization_36/FusedBatchNormV3FusedBatchNormV3conv2d_23/BiasAdd:output:0-batch_normalization_36/ReadVariableOp:value:0/batch_normalization_36/ReadVariableOp_1:value:0>batch_normalization_36/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_36/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
exponential_avg_factor%
×#<¢
%batch_normalization_36/AssignNewValueAssignVariableOp?batch_normalization_36_fusedbatchnormv3_readvariableop_resource4batch_normalization_36/FusedBatchNormV3:batch_mean:07^batch_normalization_36/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(¬
'batch_normalization_36/AssignNewValue_1AssignVariableOpAbatch_normalization_36_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_36/FusedBatchNormV3:batch_variance:09^batch_normalization_36/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(|
re_lu_31/ReluRelu+batch_normalization_36/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@­
max_pooling2d_23/MaxPoolMaxPoolre_lu_31/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
ksize
*
paddingVALID*
strides
]
dropout_31/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *«ªª?
dropout_31/dropout/MulMul!max_pooling2d_23/MaxPool:output:0!dropout_31/dropout/Const:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@i
dropout_31/dropout/ShapeShape!max_pooling2d_23/MaxPool:output:0*
T0*
_output_shapes
:ª
/dropout_31/dropout/random_uniform/RandomUniformRandomUniform!dropout_31/dropout/Shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
dtype0f
!dropout_31/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  >Ï
dropout_31/dropout/GreaterEqualGreaterEqual8dropout_31/dropout/random_uniform/RandomUniform:output:0*dropout_31/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
dropout_31/dropout/CastCast#dropout_31/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
dropout_31/dropout/Mul_1Muldropout_31/dropout/Mul:z:0dropout_31/dropout/Cast:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
conv2d_24/Conv2D/ReadVariableOpReadVariableOp(conv2d_24_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype0Ä
conv2d_24/Conv2DConv2Ddropout_31/dropout/Mul_1:z:0'conv2d_24/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides

 conv2d_24/BiasAdd/ReadVariableOpReadVariableOp)conv2d_24_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
conv2d_24/BiasAddBiasAddconv2d_24/Conv2D:output:0(conv2d_24/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%batch_normalization_37/ReadVariableOpReadVariableOp.batch_normalization_37_readvariableop_resource*
_output_shapes	
:*
dtype0
'batch_normalization_37/ReadVariableOp_1ReadVariableOp0batch_normalization_37_readvariableop_1_resource*
_output_shapes	
:*
dtype0³
6batch_normalization_37/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_37_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0·
8batch_normalization_37/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_37_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0Ð
'batch_normalization_37/FusedBatchNormV3FusedBatchNormV3conv2d_24/BiasAdd:output:0-batch_normalization_37/ReadVariableOp:value:0/batch_normalization_37/ReadVariableOp_1:value:0>batch_normalization_37/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_37/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
exponential_avg_factor%
×#<¢
%batch_normalization_37/AssignNewValueAssignVariableOp?batch_normalization_37_fusedbatchnormv3_readvariableop_resource4batch_normalization_37/FusedBatchNormV3:batch_mean:07^batch_normalization_37/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(¬
'batch_normalization_37/AssignNewValue_1AssignVariableOpAbatch_normalization_37_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_37/FusedBatchNormV3:batch_variance:09^batch_normalization_37/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
re_lu_32/ReluRelu+batch_normalization_37/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ®
max_pooling2d_24/MaxPoolMaxPoolre_lu_32/Relu:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
]
dropout_32/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *«ªª?
dropout_32/dropout/MulMul!max_pooling2d_24/MaxPool:output:0!dropout_32/dropout/Const:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
dropout_32/dropout/ShapeShape!max_pooling2d_24/MaxPool:output:0*
T0*
_output_shapes
:«
/dropout_32/dropout/random_uniform/RandomUniformRandomUniform!dropout_32/dropout/Shape:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0f
!dropout_32/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  >Ð
dropout_32/dropout/GreaterEqualGreaterEqual8dropout_32/dropout/random_uniform/RandomUniform:output:0*dropout_32/dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dropout_32/dropout/CastCast#dropout_32/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dropout_32/dropout/Mul_1Muldropout_32/dropout/Mul:z:0dropout_32/dropout/Cast:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
flatten_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ  
flatten_4/ReshapeReshapedropout_32/dropout/Mul_1:z:0flatten_4/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!features512/MatMul/ReadVariableOpReadVariableOp*features512_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0
features512/MatMulMatMulflatten_4/Reshape:output:0)features512/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"features512/BiasAdd/ReadVariableOpReadVariableOp+features512_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
features512/BiasAddBiasAddfeatures512/MatMul:product:0*features512/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
5batch_normalization_38/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: Ä
#batch_normalization_38/moments/meanMeanfeatures512/BiasAdd:output:0>batch_normalization_38/moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	*
	keep_dims(
+batch_normalization_38/moments/StopGradientStopGradient,batch_normalization_38/moments/mean:output:0*
T0*
_output_shapes
:	Ì
0batch_normalization_38/moments/SquaredDifferenceSquaredDifferencefeatures512/BiasAdd:output:04batch_normalization_38/moments/StopGradient:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
9batch_normalization_38/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ä
'batch_normalization_38/moments/varianceMean4batch_normalization_38/moments/SquaredDifference:z:0Bbatch_normalization_38/moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	*
	keep_dims(
&batch_normalization_38/moments/SqueezeSqueeze,batch_normalization_38/moments/mean:output:0*
T0*
_output_shapes	
:*
squeeze_dims
 ¢
(batch_normalization_38/moments/Squeeze_1Squeeze0batch_normalization_38/moments/variance:output:0*
T0*
_output_shapes	
:*
squeeze_dims
 q
,batch_normalization_38/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<±
5batch_normalization_38/AssignMovingAvg/ReadVariableOpReadVariableOp>batch_normalization_38_assignmovingavg_readvariableop_resource*
_output_shapes	
:*
dtype0Ç
*batch_normalization_38/AssignMovingAvg/subSub=batch_normalization_38/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_38/moments/Squeeze:output:0*
T0*
_output_shapes	
:¾
*batch_normalization_38/AssignMovingAvg/mulMul.batch_normalization_38/AssignMovingAvg/sub:z:05batch_normalization_38/AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:
&batch_normalization_38/AssignMovingAvgAssignSubVariableOp>batch_normalization_38_assignmovingavg_readvariableop_resource.batch_normalization_38/AssignMovingAvg/mul:z:06^batch_normalization_38/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0s
.batch_normalization_38/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<µ
7batch_normalization_38/AssignMovingAvg_1/ReadVariableOpReadVariableOp@batch_normalization_38_assignmovingavg_1_readvariableop_resource*
_output_shapes	
:*
dtype0Í
,batch_normalization_38/AssignMovingAvg_1/subSub?batch_normalization_38/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_38/moments/Squeeze_1:output:0*
T0*
_output_shapes	
:Ä
,batch_normalization_38/AssignMovingAvg_1/mulMul0batch_normalization_38/AssignMovingAvg_1/sub:z:07batch_normalization_38/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:
(batch_normalization_38/AssignMovingAvg_1AssignSubVariableOp@batch_normalization_38_assignmovingavg_1_readvariableop_resource0batch_normalization_38/AssignMovingAvg_1/mul:z:08^batch_normalization_38/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0k
&batch_normalization_38/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:·
$batch_normalization_38/batchnorm/addAddV21batch_normalization_38/moments/Squeeze_1:output:0/batch_normalization_38/batchnorm/add/y:output:0*
T0*
_output_shapes	
:
&batch_normalization_38/batchnorm/RsqrtRsqrt(batch_normalization_38/batchnorm/add:z:0*
T0*
_output_shapes	
:­
3batch_normalization_38/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_38_batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype0º
$batch_normalization_38/batchnorm/mulMul*batch_normalization_38/batchnorm/Rsqrt:y:0;batch_normalization_38/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:¨
&batch_normalization_38/batchnorm/mul_1Mulfeatures512/BiasAdd:output:0(batch_normalization_38/batchnorm/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ®
&batch_normalization_38/batchnorm/mul_2Mul/batch_normalization_38/moments/Squeeze:output:0(batch_normalization_38/batchnorm/mul:z:0*
T0*
_output_shapes	
:¥
/batch_normalization_38/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_38_batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype0¶
$batch_normalization_38/batchnorm/subSub7batch_normalization_38/batchnorm/ReadVariableOp:value:0*batch_normalization_38/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:¸
&batch_normalization_38/batchnorm/add_1AddV2*batch_normalization_38/batchnorm/mul_1:z:0(batch_normalization_38/batchnorm/sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿt
re_lu_33/ReluRelu*batch_normalization_38/batchnorm/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
dropout_33/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *«ªª?
dropout_33/dropout/MulMulre_lu_33/Relu:activations:0!dropout_33/dropout/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
dropout_33/dropout/ShapeShapere_lu_33/Relu:activations:0*
T0*
_output_shapes
:£
/dropout_33/dropout/random_uniform/RandomUniformRandomUniform!dropout_33/dropout/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0f
!dropout_33/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  >È
dropout_33/dropout/GreaterEqualGreaterEqual8dropout_33/dropout/random_uniform/RandomUniform:output:0*dropout_33/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dropout_33/dropout/CastCast#dropout_33/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dropout_33/dropout/Mul_1Muldropout_33/dropout/Mul:z:0dropout_33/dropout/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!features256/MatMul/ReadVariableOpReadVariableOp*features256_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0
features256/MatMulMatMuldropout_33/dropout/Mul_1:z:0)features256/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"features256/BiasAdd/ReadVariableOpReadVariableOp+features256_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
features256/BiasAddBiasAddfeatures256/MatMul:product:0*features256/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
5batch_normalization_39/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: Ä
#batch_normalization_39/moments/meanMeanfeatures256/BiasAdd:output:0>batch_normalization_39/moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	*
	keep_dims(
+batch_normalization_39/moments/StopGradientStopGradient,batch_normalization_39/moments/mean:output:0*
T0*
_output_shapes
:	Ì
0batch_normalization_39/moments/SquaredDifferenceSquaredDifferencefeatures256/BiasAdd:output:04batch_normalization_39/moments/StopGradient:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
9batch_normalization_39/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ä
'batch_normalization_39/moments/varianceMean4batch_normalization_39/moments/SquaredDifference:z:0Bbatch_normalization_39/moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	*
	keep_dims(
&batch_normalization_39/moments/SqueezeSqueeze,batch_normalization_39/moments/mean:output:0*
T0*
_output_shapes	
:*
squeeze_dims
 ¢
(batch_normalization_39/moments/Squeeze_1Squeeze0batch_normalization_39/moments/variance:output:0*
T0*
_output_shapes	
:*
squeeze_dims
 q
,batch_normalization_39/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<±
5batch_normalization_39/AssignMovingAvg/ReadVariableOpReadVariableOp>batch_normalization_39_assignmovingavg_readvariableop_resource*
_output_shapes	
:*
dtype0Ç
*batch_normalization_39/AssignMovingAvg/subSub=batch_normalization_39/AssignMovingAvg/ReadVariableOp:value:0/batch_normalization_39/moments/Squeeze:output:0*
T0*
_output_shapes	
:¾
*batch_normalization_39/AssignMovingAvg/mulMul.batch_normalization_39/AssignMovingAvg/sub:z:05batch_normalization_39/AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:
&batch_normalization_39/AssignMovingAvgAssignSubVariableOp>batch_normalization_39_assignmovingavg_readvariableop_resource.batch_normalization_39/AssignMovingAvg/mul:z:06^batch_normalization_39/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0s
.batch_normalization_39/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<µ
7batch_normalization_39/AssignMovingAvg_1/ReadVariableOpReadVariableOp@batch_normalization_39_assignmovingavg_1_readvariableop_resource*
_output_shapes	
:*
dtype0Í
,batch_normalization_39/AssignMovingAvg_1/subSub?batch_normalization_39/AssignMovingAvg_1/ReadVariableOp:value:01batch_normalization_39/moments/Squeeze_1:output:0*
T0*
_output_shapes	
:Ä
,batch_normalization_39/AssignMovingAvg_1/mulMul0batch_normalization_39/AssignMovingAvg_1/sub:z:07batch_normalization_39/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:
(batch_normalization_39/AssignMovingAvg_1AssignSubVariableOp@batch_normalization_39_assignmovingavg_1_readvariableop_resource0batch_normalization_39/AssignMovingAvg_1/mul:z:08^batch_normalization_39/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0k
&batch_normalization_39/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:·
$batch_normalization_39/batchnorm/addAddV21batch_normalization_39/moments/Squeeze_1:output:0/batch_normalization_39/batchnorm/add/y:output:0*
T0*
_output_shapes	
:
&batch_normalization_39/batchnorm/RsqrtRsqrt(batch_normalization_39/batchnorm/add:z:0*
T0*
_output_shapes	
:­
3batch_normalization_39/batchnorm/mul/ReadVariableOpReadVariableOp<batch_normalization_39_batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype0º
$batch_normalization_39/batchnorm/mulMul*batch_normalization_39/batchnorm/Rsqrt:y:0;batch_normalization_39/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:¨
&batch_normalization_39/batchnorm/mul_1Mulfeatures256/BiasAdd:output:0(batch_normalization_39/batchnorm/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ®
&batch_normalization_39/batchnorm/mul_2Mul/batch_normalization_39/moments/Squeeze:output:0(batch_normalization_39/batchnorm/mul:z:0*
T0*
_output_shapes	
:¥
/batch_normalization_39/batchnorm/ReadVariableOpReadVariableOp8batch_normalization_39_batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype0¶
$batch_normalization_39/batchnorm/subSub7batch_normalization_39/batchnorm/ReadVariableOp:value:0*batch_normalization_39/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:¸
&batch_normalization_39/batchnorm/add_1AddV2*batch_normalization_39/batchnorm/mul_1:z:0(batch_normalization_39/batchnorm/sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿt
re_lu_34/ReluRelu*batch_normalization_39/batchnorm/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
dropout_34/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *«ªª?
dropout_34/dropout/MulMulre_lu_34/Relu:activations:0!dropout_34/dropout/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
dropout_34/dropout/ShapeShapere_lu_34/Relu:activations:0*
T0*
_output_shapes
:£
/dropout_34/dropout/random_uniform/RandomUniformRandomUniform!dropout_34/dropout/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0f
!dropout_34/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  >È
dropout_34/dropout/GreaterEqualGreaterEqual8dropout_34/dropout/random_uniform/RandomUniform:output:0*dropout_34/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dropout_34/dropout/CastCast#dropout_34/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dropout_34/dropout/Mul_1Muldropout_34/dropout/Mul:z:0dropout_34/dropout/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_4/MatMul/ReadVariableOpReadVariableOp&dense_4_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0
dense_4/MatMulMatMuldropout_34/dropout/Mul_1:z:0%dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_4/BiasAddBiasAdddense_4/MatMul:product:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
dense_4/SoftmaxSoftmaxdense_4/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
IdentityIdentitydense_4/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÞ
NoOpNoOp&^batch_normalization_32/AssignNewValue(^batch_normalization_32/AssignNewValue_17^batch_normalization_32/FusedBatchNormV3/ReadVariableOp9^batch_normalization_32/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_32/ReadVariableOp(^batch_normalization_32/ReadVariableOp_1&^batch_normalization_33/AssignNewValue(^batch_normalization_33/AssignNewValue_17^batch_normalization_33/FusedBatchNormV3/ReadVariableOp9^batch_normalization_33/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_33/ReadVariableOp(^batch_normalization_33/ReadVariableOp_1&^batch_normalization_34/AssignNewValue(^batch_normalization_34/AssignNewValue_17^batch_normalization_34/FusedBatchNormV3/ReadVariableOp9^batch_normalization_34/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_34/ReadVariableOp(^batch_normalization_34/ReadVariableOp_1&^batch_normalization_35/AssignNewValue(^batch_normalization_35/AssignNewValue_17^batch_normalization_35/FusedBatchNormV3/ReadVariableOp9^batch_normalization_35/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_35/ReadVariableOp(^batch_normalization_35/ReadVariableOp_1&^batch_normalization_36/AssignNewValue(^batch_normalization_36/AssignNewValue_17^batch_normalization_36/FusedBatchNormV3/ReadVariableOp9^batch_normalization_36/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_36/ReadVariableOp(^batch_normalization_36/ReadVariableOp_1&^batch_normalization_37/AssignNewValue(^batch_normalization_37/AssignNewValue_17^batch_normalization_37/FusedBatchNormV3/ReadVariableOp9^batch_normalization_37/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_37/ReadVariableOp(^batch_normalization_37/ReadVariableOp_1'^batch_normalization_38/AssignMovingAvg6^batch_normalization_38/AssignMovingAvg/ReadVariableOp)^batch_normalization_38/AssignMovingAvg_18^batch_normalization_38/AssignMovingAvg_1/ReadVariableOp0^batch_normalization_38/batchnorm/ReadVariableOp4^batch_normalization_38/batchnorm/mul/ReadVariableOp'^batch_normalization_39/AssignMovingAvg6^batch_normalization_39/AssignMovingAvg/ReadVariableOp)^batch_normalization_39/AssignMovingAvg_18^batch_normalization_39/AssignMovingAvg_1/ReadVariableOp0^batch_normalization_39/batchnorm/ReadVariableOp4^batch_normalization_39/batchnorm/mul/ReadVariableOp!^conv2d_20/BiasAdd/ReadVariableOp ^conv2d_20/Conv2D/ReadVariableOp!^conv2d_21/BiasAdd/ReadVariableOp ^conv2d_21/Conv2D/ReadVariableOp!^conv2d_22/BiasAdd/ReadVariableOp ^conv2d_22/Conv2D/ReadVariableOp!^conv2d_23/BiasAdd/ReadVariableOp ^conv2d_23/Conv2D/ReadVariableOp!^conv2d_24/BiasAdd/ReadVariableOp ^conv2d_24/Conv2D/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp#^features256/BiasAdd/ReadVariableOp"^features256/MatMul/ReadVariableOp#^features512/BiasAdd/ReadVariableOp"^features512/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapesy
w:ÿÿÿÿÿÿÿÿÿc(: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2N
%batch_normalization_32/AssignNewValue%batch_normalization_32/AssignNewValue2R
'batch_normalization_32/AssignNewValue_1'batch_normalization_32/AssignNewValue_12p
6batch_normalization_32/FusedBatchNormV3/ReadVariableOp6batch_normalization_32/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_32/FusedBatchNormV3/ReadVariableOp_18batch_normalization_32/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_32/ReadVariableOp%batch_normalization_32/ReadVariableOp2R
'batch_normalization_32/ReadVariableOp_1'batch_normalization_32/ReadVariableOp_12N
%batch_normalization_33/AssignNewValue%batch_normalization_33/AssignNewValue2R
'batch_normalization_33/AssignNewValue_1'batch_normalization_33/AssignNewValue_12p
6batch_normalization_33/FusedBatchNormV3/ReadVariableOp6batch_normalization_33/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_33/FusedBatchNormV3/ReadVariableOp_18batch_normalization_33/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_33/ReadVariableOp%batch_normalization_33/ReadVariableOp2R
'batch_normalization_33/ReadVariableOp_1'batch_normalization_33/ReadVariableOp_12N
%batch_normalization_34/AssignNewValue%batch_normalization_34/AssignNewValue2R
'batch_normalization_34/AssignNewValue_1'batch_normalization_34/AssignNewValue_12p
6batch_normalization_34/FusedBatchNormV3/ReadVariableOp6batch_normalization_34/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_34/FusedBatchNormV3/ReadVariableOp_18batch_normalization_34/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_34/ReadVariableOp%batch_normalization_34/ReadVariableOp2R
'batch_normalization_34/ReadVariableOp_1'batch_normalization_34/ReadVariableOp_12N
%batch_normalization_35/AssignNewValue%batch_normalization_35/AssignNewValue2R
'batch_normalization_35/AssignNewValue_1'batch_normalization_35/AssignNewValue_12p
6batch_normalization_35/FusedBatchNormV3/ReadVariableOp6batch_normalization_35/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_35/FusedBatchNormV3/ReadVariableOp_18batch_normalization_35/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_35/ReadVariableOp%batch_normalization_35/ReadVariableOp2R
'batch_normalization_35/ReadVariableOp_1'batch_normalization_35/ReadVariableOp_12N
%batch_normalization_36/AssignNewValue%batch_normalization_36/AssignNewValue2R
'batch_normalization_36/AssignNewValue_1'batch_normalization_36/AssignNewValue_12p
6batch_normalization_36/FusedBatchNormV3/ReadVariableOp6batch_normalization_36/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_36/FusedBatchNormV3/ReadVariableOp_18batch_normalization_36/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_36/ReadVariableOp%batch_normalization_36/ReadVariableOp2R
'batch_normalization_36/ReadVariableOp_1'batch_normalization_36/ReadVariableOp_12N
%batch_normalization_37/AssignNewValue%batch_normalization_37/AssignNewValue2R
'batch_normalization_37/AssignNewValue_1'batch_normalization_37/AssignNewValue_12p
6batch_normalization_37/FusedBatchNormV3/ReadVariableOp6batch_normalization_37/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_37/FusedBatchNormV3/ReadVariableOp_18batch_normalization_37/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_37/ReadVariableOp%batch_normalization_37/ReadVariableOp2R
'batch_normalization_37/ReadVariableOp_1'batch_normalization_37/ReadVariableOp_12P
&batch_normalization_38/AssignMovingAvg&batch_normalization_38/AssignMovingAvg2n
5batch_normalization_38/AssignMovingAvg/ReadVariableOp5batch_normalization_38/AssignMovingAvg/ReadVariableOp2T
(batch_normalization_38/AssignMovingAvg_1(batch_normalization_38/AssignMovingAvg_12r
7batch_normalization_38/AssignMovingAvg_1/ReadVariableOp7batch_normalization_38/AssignMovingAvg_1/ReadVariableOp2b
/batch_normalization_38/batchnorm/ReadVariableOp/batch_normalization_38/batchnorm/ReadVariableOp2j
3batch_normalization_38/batchnorm/mul/ReadVariableOp3batch_normalization_38/batchnorm/mul/ReadVariableOp2P
&batch_normalization_39/AssignMovingAvg&batch_normalization_39/AssignMovingAvg2n
5batch_normalization_39/AssignMovingAvg/ReadVariableOp5batch_normalization_39/AssignMovingAvg/ReadVariableOp2T
(batch_normalization_39/AssignMovingAvg_1(batch_normalization_39/AssignMovingAvg_12r
7batch_normalization_39/AssignMovingAvg_1/ReadVariableOp7batch_normalization_39/AssignMovingAvg_1/ReadVariableOp2b
/batch_normalization_39/batchnorm/ReadVariableOp/batch_normalization_39/batchnorm/ReadVariableOp2j
3batch_normalization_39/batchnorm/mul/ReadVariableOp3batch_normalization_39/batchnorm/mul/ReadVariableOp2D
 conv2d_20/BiasAdd/ReadVariableOp conv2d_20/BiasAdd/ReadVariableOp2B
conv2d_20/Conv2D/ReadVariableOpconv2d_20/Conv2D/ReadVariableOp2D
 conv2d_21/BiasAdd/ReadVariableOp conv2d_21/BiasAdd/ReadVariableOp2B
conv2d_21/Conv2D/ReadVariableOpconv2d_21/Conv2D/ReadVariableOp2D
 conv2d_22/BiasAdd/ReadVariableOp conv2d_22/BiasAdd/ReadVariableOp2B
conv2d_22/Conv2D/ReadVariableOpconv2d_22/Conv2D/ReadVariableOp2D
 conv2d_23/BiasAdd/ReadVariableOp conv2d_23/BiasAdd/ReadVariableOp2B
conv2d_23/Conv2D/ReadVariableOpconv2d_23/Conv2D/ReadVariableOp2D
 conv2d_24/BiasAdd/ReadVariableOp conv2d_24/BiasAdd/ReadVariableOp2B
conv2d_24/Conv2D/ReadVariableOpconv2d_24/Conv2D/ReadVariableOp2@
dense_4/BiasAdd/ReadVariableOpdense_4/BiasAdd/ReadVariableOp2>
dense_4/MatMul/ReadVariableOpdense_4/MatMul/ReadVariableOp2H
"features256/BiasAdd/ReadVariableOp"features256/BiasAdd/ReadVariableOp2F
!features256/MatMul/ReadVariableOp!features256/MatMul/ReadVariableOp2H
"features512/BiasAdd/ReadVariableOp"features512/BiasAdd/ReadVariableOp2F
!features512/MatMul/ReadVariableOp!features512/MatMul/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿc(
 
_user_specified_nameinputs
Â
F
*__inference_dropout_28_layer_call_fn_18206

inputs
identity»
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_28_layer_call_and_return_conditional_losses_15863h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ1:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
 
_user_specified_nameinputs
§

ý
D__inference_conv2d_23_layer_call_and_return_conditional_losses_18503

inputs8
conv2d_readvariableop_resource: @-
biasadd_readvariableop_resource:@
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
»
L
0__inference_max_pooling2d_20_layer_call_fn_18196

inputs
identityÜ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_max_pooling2d_20_layer_call_and_return_conditional_losses_15322
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¸
E
)__inference_reshape_4_layer_call_fn_18024

inputs
identityº
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿc(* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_reshape_4_layer_call_and_return_conditional_losses_15814h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿc("
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿc(:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿc(
 
_user_specified_nameinputs
Ì

Q__inference_batch_normalization_33_layer_call_and_return_conditional_losses_15271

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0È
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ°
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
È
`
D__inference_flatten_4_layer_call_and_return_conditional_losses_16031

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ  ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
û	
d
E__inference_dropout_34_layer_call_and_return_conditional_losses_19023

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *«ªª?e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿC
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  >§
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿp
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿj
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
§

,__inference_sequential_4_layer_call_fn_17441

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:#
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:#
	unknown_9:

unknown_10:

unknown_11:

unknown_12:

unknown_13:

unknown_14:$

unknown_15: 

unknown_16: 

unknown_17: 

unknown_18: 

unknown_19: 

unknown_20: $

unknown_21: @

unknown_22:@

unknown_23:@

unknown_24:@

unknown_25:@

unknown_26:@%

unknown_27:@

unknown_28:	

unknown_29:	

unknown_30:	

unknown_31:	

unknown_32:	

unknown_33:


unknown_34:	

unknown_35:	

unknown_36:	

unknown_37:	

unknown_38:	

unknown_39:


unknown_40:	

unknown_41:	

unknown_42:	

unknown_43:	

unknown_44:	

unknown_45:	

unknown_46:
identity¢StatefulPartitionedCallÚ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46*<
Tin5
321*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*R
_read_only_resource_inputs4
20	
 !"#$%&'()*+,-./0*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_sequential_4_layer_call_and_return_conditional_losses_16129o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapesy
w:ÿÿÿÿÿÿÿÿÿc(: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿc(
 
_user_specified_nameinputs
³
Õ
6__inference_batch_normalization_38_layer_call_fn_18783

inputs
unknown:	
	unknown_0:	
	unknown_1:	
	unknown_2:	
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_batch_normalization_38_layer_call_and_return_conditional_losses_15653p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
³

d
E__inference_dropout_30_layer_call_and_return_conditional_losses_16420

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *«ªª?l
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  >®
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ w
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ q
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ a
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ :W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
¾
D
(__inference_re_lu_31_layer_call_fn_18570

inputs
identity¹
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_re_lu_31_layer_call_and_return_conditional_losses_15975h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
	
Ñ
6__inference_batch_normalization_34_layer_call_fn_18273

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_batch_normalization_34_layer_call_and_return_conditional_losses_15378
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

g
K__inference_max_pooling2d_24_layer_call_and_return_conditional_losses_18713

inputs
identity¢
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ò«
í
G__inference_sequential_4_layer_call_and_return_conditional_losses_17237
reshape_4_input*
batch_normalization_32_17104:*
batch_normalization_32_17106:*
batch_normalization_32_17108:*
batch_normalization_32_17110:)
conv2d_20_17113:
conv2d_20_17115:*
batch_normalization_33_17118:*
batch_normalization_33_17120:*
batch_normalization_33_17122:*
batch_normalization_33_17124:)
conv2d_21_17130:
conv2d_21_17132:*
batch_normalization_34_17135:*
batch_normalization_34_17137:*
batch_normalization_34_17139:*
batch_normalization_34_17141:)
conv2d_22_17147: 
conv2d_22_17149: *
batch_normalization_35_17152: *
batch_normalization_35_17154: *
batch_normalization_35_17156: *
batch_normalization_35_17158: )
conv2d_23_17164: @
conv2d_23_17166:@*
batch_normalization_36_17169:@*
batch_normalization_36_17171:@*
batch_normalization_36_17173:@*
batch_normalization_36_17175:@*
conv2d_24_17181:@
conv2d_24_17183:	+
batch_normalization_37_17186:	+
batch_normalization_37_17188:	+
batch_normalization_37_17190:	+
batch_normalization_37_17192:	%
features512_17199:
 
features512_17201:	+
batch_normalization_38_17204:	+
batch_normalization_38_17206:	+
batch_normalization_38_17208:	+
batch_normalization_38_17210:	%
features256_17215:
 
features256_17217:	+
batch_normalization_39_17220:	+
batch_normalization_39_17222:	+
batch_normalization_39_17224:	+
batch_normalization_39_17226:	 
dense_4_17231:	
dense_4_17233:
identity¢.batch_normalization_32/StatefulPartitionedCall¢.batch_normalization_33/StatefulPartitionedCall¢.batch_normalization_34/StatefulPartitionedCall¢.batch_normalization_35/StatefulPartitionedCall¢.batch_normalization_36/StatefulPartitionedCall¢.batch_normalization_37/StatefulPartitionedCall¢.batch_normalization_38/StatefulPartitionedCall¢.batch_normalization_39/StatefulPartitionedCall¢!conv2d_20/StatefulPartitionedCall¢!conv2d_21/StatefulPartitionedCall¢!conv2d_22/StatefulPartitionedCall¢!conv2d_23/StatefulPartitionedCall¢!conv2d_24/StatefulPartitionedCall¢dense_4/StatefulPartitionedCall¢"dropout_28/StatefulPartitionedCall¢"dropout_29/StatefulPartitionedCall¢"dropout_30/StatefulPartitionedCall¢"dropout_31/StatefulPartitionedCall¢"dropout_32/StatefulPartitionedCall¢"dropout_33/StatefulPartitionedCall¢"dropout_34/StatefulPartitionedCall¢#features256/StatefulPartitionedCall¢#features512/StatefulPartitionedCallÍ
reshape_4/PartitionedCallPartitionedCallreshape_4_input*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿc(* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_reshape_4_layer_call_and_return_conditional_losses_15814
.batch_normalization_32/StatefulPartitionedCallStatefulPartitionedCall"reshape_4/PartitionedCall:output:0batch_normalization_32_17104batch_normalization_32_17106batch_normalization_32_17108batch_normalization_32_17110*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿc(*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_batch_normalization_32_layer_call_and_return_conditional_losses_15238­
!conv2d_20/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_32/StatefulPartitionedCall:output:0conv2d_20_17113conv2d_20_17115*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿc(*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_20_layer_call_and_return_conditional_losses_15835
.batch_normalization_33/StatefulPartitionedCallStatefulPartitionedCall*conv2d_20/StatefulPartitionedCall:output:0batch_normalization_33_17118batch_normalization_33_17120batch_normalization_33_17122batch_normalization_33_17124*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿc(*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_batch_normalization_33_layer_call_and_return_conditional_losses_15302ó
re_lu_28/PartitionedCallPartitionedCall7batch_normalization_33/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿc(* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_re_lu_28_layer_call_and_return_conditional_losses_15855í
 max_pooling2d_20/PartitionedCallPartitionedCall!re_lu_28/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_max_pooling2d_20_layer_call_and_return_conditional_losses_15322ù
"dropout_28/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_20/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_28_layer_call_and_return_conditional_losses_16498¡
!conv2d_21/StatefulPartitionedCallStatefulPartitionedCall+dropout_28/StatefulPartitionedCall:output:0conv2d_21_17130conv2d_21_17132*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_21_layer_call_and_return_conditional_losses_15875
.batch_normalization_34/StatefulPartitionedCallStatefulPartitionedCall*conv2d_21/StatefulPartitionedCall:output:0batch_normalization_34_17135batch_normalization_34_17137batch_normalization_34_17139batch_normalization_34_17141*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_batch_normalization_34_layer_call_and_return_conditional_losses_15378ó
re_lu_29/PartitionedCallPartitionedCall7batch_normalization_34/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_re_lu_29_layer_call_and_return_conditional_losses_15895í
 max_pooling2d_21/PartitionedCallPartitionedCall!re_lu_29/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_max_pooling2d_21_layer_call_and_return_conditional_losses_15398
"dropout_29/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_21/PartitionedCall:output:0#^dropout_28/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_29_layer_call_and_return_conditional_losses_16459¡
!conv2d_22/StatefulPartitionedCallStatefulPartitionedCall+dropout_29/StatefulPartitionedCall:output:0conv2d_22_17147conv2d_22_17149*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_22_layer_call_and_return_conditional_losses_15915
.batch_normalization_35/StatefulPartitionedCallStatefulPartitionedCall*conv2d_22/StatefulPartitionedCall:output:0batch_normalization_35_17152batch_normalization_35_17154batch_normalization_35_17156batch_normalization_35_17158*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_batch_normalization_35_layer_call_and_return_conditional_losses_15454ó
re_lu_30/PartitionedCallPartitionedCall7batch_normalization_35/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_re_lu_30_layer_call_and_return_conditional_losses_15935í
 max_pooling2d_22/PartitionedCallPartitionedCall!re_lu_30/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_max_pooling2d_22_layer_call_and_return_conditional_losses_15474
"dropout_30/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_22/PartitionedCall:output:0#^dropout_29/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_30_layer_call_and_return_conditional_losses_16420¡
!conv2d_23/StatefulPartitionedCallStatefulPartitionedCall+dropout_30/StatefulPartitionedCall:output:0conv2d_23_17164conv2d_23_17166*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_23_layer_call_and_return_conditional_losses_15955
.batch_normalization_36/StatefulPartitionedCallStatefulPartitionedCall*conv2d_23/StatefulPartitionedCall:output:0batch_normalization_36_17169batch_normalization_36_17171batch_normalization_36_17173batch_normalization_36_17175*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_batch_normalization_36_layer_call_and_return_conditional_losses_15530ó
re_lu_31/PartitionedCallPartitionedCall7batch_normalization_36/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_re_lu_31_layer_call_and_return_conditional_losses_15975í
 max_pooling2d_23/PartitionedCallPartitionedCall!re_lu_31/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_max_pooling2d_23_layer_call_and_return_conditional_losses_15550
"dropout_31/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_23/PartitionedCall:output:0#^dropout_30/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_31_layer_call_and_return_conditional_losses_16381¢
!conv2d_24/StatefulPartitionedCallStatefulPartitionedCall+dropout_31/StatefulPartitionedCall:output:0conv2d_24_17181conv2d_24_17183*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_24_layer_call_and_return_conditional_losses_15995
.batch_normalization_37/StatefulPartitionedCallStatefulPartitionedCall*conv2d_24/StatefulPartitionedCall:output:0batch_normalization_37_17186batch_normalization_37_17188batch_normalization_37_17190batch_normalization_37_17192*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_batch_normalization_37_layer_call_and_return_conditional_losses_15606ô
re_lu_32/PartitionedCallPartitionedCall7batch_normalization_37/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_re_lu_32_layer_call_and_return_conditional_losses_16015î
 max_pooling2d_24/PartitionedCallPartitionedCall!re_lu_32/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_max_pooling2d_24_layer_call_and_return_conditional_losses_15626
"dropout_32/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_24/PartitionedCall:output:0#^dropout_31/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_32_layer_call_and_return_conditional_losses_16342â
flatten_4/PartitionedCallPartitionedCall+dropout_32/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_flatten_4_layer_call_and_return_conditional_losses_16031
#features512/StatefulPartitionedCallStatefulPartitionedCall"flatten_4/PartitionedCall:output:0features512_17199features512_17201*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_features512_layer_call_and_return_conditional_losses_16043
.batch_normalization_38/StatefulPartitionedCallStatefulPartitionedCall,features512/StatefulPartitionedCall:output:0batch_normalization_38_17204batch_normalization_38_17206batch_normalization_38_17208batch_normalization_38_17210*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_batch_normalization_38_layer_call_and_return_conditional_losses_15700ì
re_lu_33/PartitionedCallPartitionedCall7batch_normalization_38/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_re_lu_33_layer_call_and_return_conditional_losses_16063
"dropout_33/StatefulPartitionedCallStatefulPartitionedCall!re_lu_33/PartitionedCall:output:0#^dropout_32/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_33_layer_call_and_return_conditional_losses_16297¢
#features256/StatefulPartitionedCallStatefulPartitionedCall+dropout_33/StatefulPartitionedCall:output:0features256_17215features256_17217*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_features256_layer_call_and_return_conditional_losses_16082
.batch_normalization_39/StatefulPartitionedCallStatefulPartitionedCall,features256/StatefulPartitionedCall:output:0batch_normalization_39_17220batch_normalization_39_17222batch_normalization_39_17224batch_normalization_39_17226*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_batch_normalization_39_layer_call_and_return_conditional_losses_15782ì
re_lu_34/PartitionedCallPartitionedCall7batch_normalization_39/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_re_lu_34_layer_call_and_return_conditional_losses_16102
"dropout_34/StatefulPartitionedCallStatefulPartitionedCall!re_lu_34/PartitionedCall:output:0#^dropout_33/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_34_layer_call_and_return_conditional_losses_16258
dense_4/StatefulPartitionedCallStatefulPartitionedCall+dropout_34/StatefulPartitionedCall:output:0dense_4_17231dense_4_17233*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_4_layer_call_and_return_conditional_losses_16122w
IdentityIdentity(dense_4/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿó
NoOpNoOp/^batch_normalization_32/StatefulPartitionedCall/^batch_normalization_33/StatefulPartitionedCall/^batch_normalization_34/StatefulPartitionedCall/^batch_normalization_35/StatefulPartitionedCall/^batch_normalization_36/StatefulPartitionedCall/^batch_normalization_37/StatefulPartitionedCall/^batch_normalization_38/StatefulPartitionedCall/^batch_normalization_39/StatefulPartitionedCall"^conv2d_20/StatefulPartitionedCall"^conv2d_21/StatefulPartitionedCall"^conv2d_22/StatefulPartitionedCall"^conv2d_23/StatefulPartitionedCall"^conv2d_24/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall#^dropout_28/StatefulPartitionedCall#^dropout_29/StatefulPartitionedCall#^dropout_30/StatefulPartitionedCall#^dropout_31/StatefulPartitionedCall#^dropout_32/StatefulPartitionedCall#^dropout_33/StatefulPartitionedCall#^dropout_34/StatefulPartitionedCall$^features256/StatefulPartitionedCall$^features512/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapesy
w:ÿÿÿÿÿÿÿÿÿc(: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2`
.batch_normalization_32/StatefulPartitionedCall.batch_normalization_32/StatefulPartitionedCall2`
.batch_normalization_33/StatefulPartitionedCall.batch_normalization_33/StatefulPartitionedCall2`
.batch_normalization_34/StatefulPartitionedCall.batch_normalization_34/StatefulPartitionedCall2`
.batch_normalization_35/StatefulPartitionedCall.batch_normalization_35/StatefulPartitionedCall2`
.batch_normalization_36/StatefulPartitionedCall.batch_normalization_36/StatefulPartitionedCall2`
.batch_normalization_37/StatefulPartitionedCall.batch_normalization_37/StatefulPartitionedCall2`
.batch_normalization_38/StatefulPartitionedCall.batch_normalization_38/StatefulPartitionedCall2`
.batch_normalization_39/StatefulPartitionedCall.batch_normalization_39/StatefulPartitionedCall2F
!conv2d_20/StatefulPartitionedCall!conv2d_20/StatefulPartitionedCall2F
!conv2d_21/StatefulPartitionedCall!conv2d_21/StatefulPartitionedCall2F
!conv2d_22/StatefulPartitionedCall!conv2d_22/StatefulPartitionedCall2F
!conv2d_23/StatefulPartitionedCall!conv2d_23/StatefulPartitionedCall2F
!conv2d_24/StatefulPartitionedCall!conv2d_24/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2H
"dropout_28/StatefulPartitionedCall"dropout_28/StatefulPartitionedCall2H
"dropout_29/StatefulPartitionedCall"dropout_29/StatefulPartitionedCall2H
"dropout_30/StatefulPartitionedCall"dropout_30/StatefulPartitionedCall2H
"dropout_31/StatefulPartitionedCall"dropout_31/StatefulPartitionedCall2H
"dropout_32/StatefulPartitionedCall"dropout_32/StatefulPartitionedCall2H
"dropout_33/StatefulPartitionedCall"dropout_33/StatefulPartitionedCall2H
"dropout_34/StatefulPartitionedCall"dropout_34/StatefulPartitionedCall2J
#features256/StatefulPartitionedCall#features256/StatefulPartitionedCall2J
#features512/StatefulPartitionedCall#features512/StatefulPartitionedCall:\ X
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿc(
)
_user_specified_namereshape_4_input
à
´
Q__inference_batch_normalization_39_layer_call_and_return_conditional_losses_18952

inputs0
!batchnorm_readvariableop_resource:	4
%batchnorm_mul_readvariableop_resource:	2
#batchnorm_readvariableop_1_resource:	2
#batchnorm_readvariableop_2_resource:	
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOpw
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:x
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:Q
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype0u
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:d
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ{
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:*
dtype0s
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:{
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:*
dtype0s
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:s
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¯%
î
Q__inference_batch_normalization_39_layer_call_and_return_conditional_losses_18986

inputs6
'assignmovingavg_readvariableop_resource:	8
)assignmovingavg_1_readvariableop_resource:	4
%batchnorm_mul_readvariableop_resource:	0
!batchnorm_readvariableop_resource:	
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	*
	keep_dims(e
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	*
	keep_dims(n
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:*
squeeze_dims
 t
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes	
:*
dtype0
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes	
:y
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:¬
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes	
:*
dtype0
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes	
:
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:´
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:r
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:Q
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype0u
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:d
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:w
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype0q
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:s
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿê
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
í

)__inference_conv2d_23_layer_call_fn_18493

inputs!
unknown: @
	unknown_0:@
identity¢StatefulPartitionedCallä
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_23_layer_call_and_return_conditional_losses_15955w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
§

ý
D__inference_conv2d_20_layer_call_and_return_conditional_losses_15835

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿc(*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿc(g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿc(w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿc(: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿc(
 
_user_specified_nameinputs
Ü
c
E__inference_dropout_34_layer_call_and_return_conditional_losses_16109

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
û	
d
E__inference_dropout_34_layer_call_and_return_conditional_losses_16258

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *«ªª?e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿC
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  >§
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿp
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿj
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
§

ý
D__inference_conv2d_22_layer_call_and_return_conditional_losses_15915

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ
: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs
Ì

Q__inference_batch_normalization_36_layer_call_and_return_conditional_losses_18547

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0È
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@°
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
û	
d
E__inference_dropout_33_layer_call_and_return_conditional_losses_18887

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *«ªª?e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿC
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  >§
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿp
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿj
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ü
c
E__inference_dropout_33_layer_call_and_return_conditional_losses_18875

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ó
`
D__inference_reshape_4_layer_call_and_return_conditional_losses_15814

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :cQ
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :(Q
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :©
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿc(`
IdentityIdentityReshape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿc("
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿc(:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿc(
 
_user_specified_nameinputs
³

d
E__inference_dropout_29_layer_call_and_return_conditional_losses_16459

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *«ªª?l
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  >®
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
w
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
q
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
a
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ
:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs
Ô	
ú
F__inference_features512_layer_call_and_return_conditional_losses_16043

inputs2
matmul_readvariableop_resource:
.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
IdentityIdentityBiasAdd:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ç
_
C__inference_re_lu_28_layer_call_and_return_conditional_losses_15855

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿc(b
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿc("
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿc(:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿc(
 
_user_specified_nameinputs
à
´
Q__inference_batch_normalization_38_layer_call_and_return_conditional_losses_15653

inputs0
!batchnorm_readvariableop_resource:	4
%batchnorm_mul_readvariableop_resource:	2
#batchnorm_readvariableop_1_resource:	2
#batchnorm_readvariableop_2_resource:	
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOpw
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:x
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:Q
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype0u
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:d
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ{
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:*
dtype0s
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:{
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:*
dtype0s
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:s
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
§

ý
D__inference_conv2d_20_layer_call_and_return_conditional_losses_18119

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿc(*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿc(g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿc(w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿc(: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿc(
 
_user_specified_nameinputs
	
Ñ
6__inference_batch_normalization_36_layer_call_fn_18516

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_batch_normalization_36_layer_call_and_return_conditional_losses_15499
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
ü
c
E__inference_dropout_32_layer_call_and_return_conditional_losses_18728

inputs

identity_1W
IdentityIdentityinputs*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿd

Identity_1IdentityIdentity:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
§

ý
D__inference_conv2d_21_layer_call_and_return_conditional_losses_15875

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ1: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
 
_user_specified_nameinputs
®

ÿ
D__inference_conv2d_24_layer_call_and_return_conditional_losses_15995

inputs9
conv2d_readvariableop_resource:@.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
IdentityIdentityBiasAdd:output:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Ü
 
Q__inference_batch_normalization_37_layer_call_and_return_conditional_losses_18675

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0Í
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ°
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ô	
ú
F__inference_features256_layer_call_and_return_conditional_losses_18906

inputs2
matmul_readvariableop_resource:
.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
IdentityIdentityBiasAdd:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ë
_
C__inference_re_lu_32_layer_call_and_return_conditional_losses_16015

inputs
identityO
ReluReluinputs*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ü
 
Q__inference_batch_normalization_37_layer_call_and_return_conditional_losses_15575

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0Í
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ°
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Â

,__inference_sequential_4_layer_call_fn_16228
reshape_4_input
unknown:
	unknown_0:
	unknown_1:
	unknown_2:#
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:#
	unknown_9:

unknown_10:

unknown_11:

unknown_12:

unknown_13:

unknown_14:$

unknown_15: 

unknown_16: 

unknown_17: 

unknown_18: 

unknown_19: 

unknown_20: $

unknown_21: @

unknown_22:@

unknown_23:@

unknown_24:@

unknown_25:@

unknown_26:@%

unknown_27:@

unknown_28:	

unknown_29:	

unknown_30:	

unknown_31:	

unknown_32:	

unknown_33:


unknown_34:	

unknown_35:	

unknown_36:	

unknown_37:	

unknown_38:	

unknown_39:


unknown_40:	

unknown_41:	

unknown_42:	

unknown_43:	

unknown_44:	

unknown_45:	

unknown_46:
identity¢StatefulPartitionedCallã
StatefulPartitionedCallStatefulPartitionedCallreshape_4_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46*<
Tin5
321*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*R
_read_only_resource_inputs4
20	
 !"#$%&'()*+,-./0*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_sequential_4_layer_call_and_return_conditional_losses_16129o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapesy
w:ÿÿÿÿÿÿÿÿÿc(: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿc(
)
_user_specified_namereshape_4_input
Ì

Q__inference_batch_normalization_34_layer_call_and_return_conditional_losses_15347

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0È
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ°
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

g
K__inference_max_pooling2d_21_layer_call_and_return_conditional_losses_15398

inputs
identity¢
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¯%
î
Q__inference_batch_normalization_38_layer_call_and_return_conditional_losses_15700

inputs6
'assignmovingavg_readvariableop_resource:	8
)assignmovingavg_1_readvariableop_resource:	4
%batchnorm_mul_readvariableop_resource:	0
!batchnorm_readvariableop_resource:	
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	*
	keep_dims(e
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	*
	keep_dims(n
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:*
squeeze_dims
 t
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes	
:*
dtype0
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes	
:y
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:¬
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes	
:*
dtype0
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes	
:
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:´
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:r
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:Q
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype0u
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:d
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:w
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype0q
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:s
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿê
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ì

Q__inference_batch_normalization_35_layer_call_and_return_conditional_losses_18419

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0È
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : : :*
epsilon%o:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ °
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs

À
Q__inference_batch_normalization_33_layer_call_and_return_conditional_losses_18181

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity¢AssignNewValue¢AssignNewValue_1¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0Ö
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
exponential_avg_factor%
×#<Æ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(Ð
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÔ
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ë
_
C__inference_re_lu_32_layer_call_and_return_conditional_losses_18703

inputs
identityO
ReluReluinputs*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¦
F
*__inference_dropout_33_layer_call_fn_18865

inputs
identity´
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_33_layer_call_and_return_conditional_losses_16070a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ð

+__inference_features256_layer_call_fn_18896

inputs
unknown:

	unknown_0:	
identity¢StatefulPartitionedCallß
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_features256_layer_call_and_return_conditional_losses_16082p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¦
F
*__inference_dropout_34_layer_call_fn_19001

inputs
identity´
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_34_layer_call_and_return_conditional_losses_16109a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¢
D
(__inference_re_lu_34_layer_call_fn_18991

inputs
identity²
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_re_lu_34_layer_call_and_return_conditional_losses_16102a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
³
Õ
6__inference_batch_normalization_39_layer_call_fn_18919

inputs
unknown:	
	unknown_0:	
	unknown_1:	
	unknown_2:	
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_batch_normalization_39_layer_call_and_return_conditional_losses_15735p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

g
K__inference_max_pooling2d_22_layer_call_and_return_conditional_losses_18457

inputs
identity¢
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ä

'__inference_dense_4_layer_call_fn_19032

inputs
unknown:	
	unknown_0:
identity¢StatefulPartitionedCallÚ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_4_layer_call_and_return_conditional_losses_16122o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
§

ý
D__inference_conv2d_23_layer_call_and_return_conditional_losses_15955

inputs8
conv2d_readvariableop_resource: @-
biasadd_readvariableop_resource:@
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
·«
ä
G__inference_sequential_4_layer_call_and_return_conditional_losses_16763

inputs*
batch_normalization_32_16630:*
batch_normalization_32_16632:*
batch_normalization_32_16634:*
batch_normalization_32_16636:)
conv2d_20_16639:
conv2d_20_16641:*
batch_normalization_33_16644:*
batch_normalization_33_16646:*
batch_normalization_33_16648:*
batch_normalization_33_16650:)
conv2d_21_16656:
conv2d_21_16658:*
batch_normalization_34_16661:*
batch_normalization_34_16663:*
batch_normalization_34_16665:*
batch_normalization_34_16667:)
conv2d_22_16673: 
conv2d_22_16675: *
batch_normalization_35_16678: *
batch_normalization_35_16680: *
batch_normalization_35_16682: *
batch_normalization_35_16684: )
conv2d_23_16690: @
conv2d_23_16692:@*
batch_normalization_36_16695:@*
batch_normalization_36_16697:@*
batch_normalization_36_16699:@*
batch_normalization_36_16701:@*
conv2d_24_16707:@
conv2d_24_16709:	+
batch_normalization_37_16712:	+
batch_normalization_37_16714:	+
batch_normalization_37_16716:	+
batch_normalization_37_16718:	%
features512_16725:
 
features512_16727:	+
batch_normalization_38_16730:	+
batch_normalization_38_16732:	+
batch_normalization_38_16734:	+
batch_normalization_38_16736:	%
features256_16741:
 
features256_16743:	+
batch_normalization_39_16746:	+
batch_normalization_39_16748:	+
batch_normalization_39_16750:	+
batch_normalization_39_16752:	 
dense_4_16757:	
dense_4_16759:
identity¢.batch_normalization_32/StatefulPartitionedCall¢.batch_normalization_33/StatefulPartitionedCall¢.batch_normalization_34/StatefulPartitionedCall¢.batch_normalization_35/StatefulPartitionedCall¢.batch_normalization_36/StatefulPartitionedCall¢.batch_normalization_37/StatefulPartitionedCall¢.batch_normalization_38/StatefulPartitionedCall¢.batch_normalization_39/StatefulPartitionedCall¢!conv2d_20/StatefulPartitionedCall¢!conv2d_21/StatefulPartitionedCall¢!conv2d_22/StatefulPartitionedCall¢!conv2d_23/StatefulPartitionedCall¢!conv2d_24/StatefulPartitionedCall¢dense_4/StatefulPartitionedCall¢"dropout_28/StatefulPartitionedCall¢"dropout_29/StatefulPartitionedCall¢"dropout_30/StatefulPartitionedCall¢"dropout_31/StatefulPartitionedCall¢"dropout_32/StatefulPartitionedCall¢"dropout_33/StatefulPartitionedCall¢"dropout_34/StatefulPartitionedCall¢#features256/StatefulPartitionedCall¢#features512/StatefulPartitionedCallÄ
reshape_4/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿc(* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_reshape_4_layer_call_and_return_conditional_losses_15814
.batch_normalization_32/StatefulPartitionedCallStatefulPartitionedCall"reshape_4/PartitionedCall:output:0batch_normalization_32_16630batch_normalization_32_16632batch_normalization_32_16634batch_normalization_32_16636*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿc(*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_batch_normalization_32_layer_call_and_return_conditional_losses_15238­
!conv2d_20/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_32/StatefulPartitionedCall:output:0conv2d_20_16639conv2d_20_16641*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿc(*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_20_layer_call_and_return_conditional_losses_15835
.batch_normalization_33/StatefulPartitionedCallStatefulPartitionedCall*conv2d_20/StatefulPartitionedCall:output:0batch_normalization_33_16644batch_normalization_33_16646batch_normalization_33_16648batch_normalization_33_16650*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿc(*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_batch_normalization_33_layer_call_and_return_conditional_losses_15302ó
re_lu_28/PartitionedCallPartitionedCall7batch_normalization_33/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿc(* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_re_lu_28_layer_call_and_return_conditional_losses_15855í
 max_pooling2d_20/PartitionedCallPartitionedCall!re_lu_28/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_max_pooling2d_20_layer_call_and_return_conditional_losses_15322ù
"dropout_28/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_20/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_28_layer_call_and_return_conditional_losses_16498¡
!conv2d_21/StatefulPartitionedCallStatefulPartitionedCall+dropout_28/StatefulPartitionedCall:output:0conv2d_21_16656conv2d_21_16658*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_21_layer_call_and_return_conditional_losses_15875
.batch_normalization_34/StatefulPartitionedCallStatefulPartitionedCall*conv2d_21/StatefulPartitionedCall:output:0batch_normalization_34_16661batch_normalization_34_16663batch_normalization_34_16665batch_normalization_34_16667*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_batch_normalization_34_layer_call_and_return_conditional_losses_15378ó
re_lu_29/PartitionedCallPartitionedCall7batch_normalization_34/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_re_lu_29_layer_call_and_return_conditional_losses_15895í
 max_pooling2d_21/PartitionedCallPartitionedCall!re_lu_29/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_max_pooling2d_21_layer_call_and_return_conditional_losses_15398
"dropout_29/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_21/PartitionedCall:output:0#^dropout_28/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_29_layer_call_and_return_conditional_losses_16459¡
!conv2d_22/StatefulPartitionedCallStatefulPartitionedCall+dropout_29/StatefulPartitionedCall:output:0conv2d_22_16673conv2d_22_16675*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_22_layer_call_and_return_conditional_losses_15915
.batch_normalization_35/StatefulPartitionedCallStatefulPartitionedCall*conv2d_22/StatefulPartitionedCall:output:0batch_normalization_35_16678batch_normalization_35_16680batch_normalization_35_16682batch_normalization_35_16684*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_batch_normalization_35_layer_call_and_return_conditional_losses_15454ó
re_lu_30/PartitionedCallPartitionedCall7batch_normalization_35/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_re_lu_30_layer_call_and_return_conditional_losses_15935í
 max_pooling2d_22/PartitionedCallPartitionedCall!re_lu_30/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_max_pooling2d_22_layer_call_and_return_conditional_losses_15474
"dropout_30/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_22/PartitionedCall:output:0#^dropout_29/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_30_layer_call_and_return_conditional_losses_16420¡
!conv2d_23/StatefulPartitionedCallStatefulPartitionedCall+dropout_30/StatefulPartitionedCall:output:0conv2d_23_16690conv2d_23_16692*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_23_layer_call_and_return_conditional_losses_15955
.batch_normalization_36/StatefulPartitionedCallStatefulPartitionedCall*conv2d_23/StatefulPartitionedCall:output:0batch_normalization_36_16695batch_normalization_36_16697batch_normalization_36_16699batch_normalization_36_16701*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_batch_normalization_36_layer_call_and_return_conditional_losses_15530ó
re_lu_31/PartitionedCallPartitionedCall7batch_normalization_36/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_re_lu_31_layer_call_and_return_conditional_losses_15975í
 max_pooling2d_23/PartitionedCallPartitionedCall!re_lu_31/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_max_pooling2d_23_layer_call_and_return_conditional_losses_15550
"dropout_31/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_23/PartitionedCall:output:0#^dropout_30/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_31_layer_call_and_return_conditional_losses_16381¢
!conv2d_24/StatefulPartitionedCallStatefulPartitionedCall+dropout_31/StatefulPartitionedCall:output:0conv2d_24_16707conv2d_24_16709*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_24_layer_call_and_return_conditional_losses_15995
.batch_normalization_37/StatefulPartitionedCallStatefulPartitionedCall*conv2d_24/StatefulPartitionedCall:output:0batch_normalization_37_16712batch_normalization_37_16714batch_normalization_37_16716batch_normalization_37_16718*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_batch_normalization_37_layer_call_and_return_conditional_losses_15606ô
re_lu_32/PartitionedCallPartitionedCall7batch_normalization_37/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_re_lu_32_layer_call_and_return_conditional_losses_16015î
 max_pooling2d_24/PartitionedCallPartitionedCall!re_lu_32/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_max_pooling2d_24_layer_call_and_return_conditional_losses_15626
"dropout_32/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_24/PartitionedCall:output:0#^dropout_31/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_32_layer_call_and_return_conditional_losses_16342â
flatten_4/PartitionedCallPartitionedCall+dropout_32/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_flatten_4_layer_call_and_return_conditional_losses_16031
#features512/StatefulPartitionedCallStatefulPartitionedCall"flatten_4/PartitionedCall:output:0features512_16725features512_16727*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_features512_layer_call_and_return_conditional_losses_16043
.batch_normalization_38/StatefulPartitionedCallStatefulPartitionedCall,features512/StatefulPartitionedCall:output:0batch_normalization_38_16730batch_normalization_38_16732batch_normalization_38_16734batch_normalization_38_16736*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_batch_normalization_38_layer_call_and_return_conditional_losses_15700ì
re_lu_33/PartitionedCallPartitionedCall7batch_normalization_38/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_re_lu_33_layer_call_and_return_conditional_losses_16063
"dropout_33/StatefulPartitionedCallStatefulPartitionedCall!re_lu_33/PartitionedCall:output:0#^dropout_32/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_33_layer_call_and_return_conditional_losses_16297¢
#features256/StatefulPartitionedCallStatefulPartitionedCall+dropout_33/StatefulPartitionedCall:output:0features256_16741features256_16743*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_features256_layer_call_and_return_conditional_losses_16082
.batch_normalization_39/StatefulPartitionedCallStatefulPartitionedCall,features256/StatefulPartitionedCall:output:0batch_normalization_39_16746batch_normalization_39_16748batch_normalization_39_16750batch_normalization_39_16752*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_batch_normalization_39_layer_call_and_return_conditional_losses_15782ì
re_lu_34/PartitionedCallPartitionedCall7batch_normalization_39/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_re_lu_34_layer_call_and_return_conditional_losses_16102
"dropout_34/StatefulPartitionedCallStatefulPartitionedCall!re_lu_34/PartitionedCall:output:0#^dropout_33/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_34_layer_call_and_return_conditional_losses_16258
dense_4/StatefulPartitionedCallStatefulPartitionedCall+dropout_34/StatefulPartitionedCall:output:0dense_4_16757dense_4_16759*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_4_layer_call_and_return_conditional_losses_16122w
IdentityIdentity(dense_4/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿó
NoOpNoOp/^batch_normalization_32/StatefulPartitionedCall/^batch_normalization_33/StatefulPartitionedCall/^batch_normalization_34/StatefulPartitionedCall/^batch_normalization_35/StatefulPartitionedCall/^batch_normalization_36/StatefulPartitionedCall/^batch_normalization_37/StatefulPartitionedCall/^batch_normalization_38/StatefulPartitionedCall/^batch_normalization_39/StatefulPartitionedCall"^conv2d_20/StatefulPartitionedCall"^conv2d_21/StatefulPartitionedCall"^conv2d_22/StatefulPartitionedCall"^conv2d_23/StatefulPartitionedCall"^conv2d_24/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall#^dropout_28/StatefulPartitionedCall#^dropout_29/StatefulPartitionedCall#^dropout_30/StatefulPartitionedCall#^dropout_31/StatefulPartitionedCall#^dropout_32/StatefulPartitionedCall#^dropout_33/StatefulPartitionedCall#^dropout_34/StatefulPartitionedCall$^features256/StatefulPartitionedCall$^features512/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapesy
w:ÿÿÿÿÿÿÿÿÿc(: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2`
.batch_normalization_32/StatefulPartitionedCall.batch_normalization_32/StatefulPartitionedCall2`
.batch_normalization_33/StatefulPartitionedCall.batch_normalization_33/StatefulPartitionedCall2`
.batch_normalization_34/StatefulPartitionedCall.batch_normalization_34/StatefulPartitionedCall2`
.batch_normalization_35/StatefulPartitionedCall.batch_normalization_35/StatefulPartitionedCall2`
.batch_normalization_36/StatefulPartitionedCall.batch_normalization_36/StatefulPartitionedCall2`
.batch_normalization_37/StatefulPartitionedCall.batch_normalization_37/StatefulPartitionedCall2`
.batch_normalization_38/StatefulPartitionedCall.batch_normalization_38/StatefulPartitionedCall2`
.batch_normalization_39/StatefulPartitionedCall.batch_normalization_39/StatefulPartitionedCall2F
!conv2d_20/StatefulPartitionedCall!conv2d_20/StatefulPartitionedCall2F
!conv2d_21/StatefulPartitionedCall!conv2d_21/StatefulPartitionedCall2F
!conv2d_22/StatefulPartitionedCall!conv2d_22/StatefulPartitionedCall2F
!conv2d_23/StatefulPartitionedCall!conv2d_23/StatefulPartitionedCall2F
!conv2d_24/StatefulPartitionedCall!conv2d_24/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2H
"dropout_28/StatefulPartitionedCall"dropout_28/StatefulPartitionedCall2H
"dropout_29/StatefulPartitionedCall"dropout_29/StatefulPartitionedCall2H
"dropout_30/StatefulPartitionedCall"dropout_30/StatefulPartitionedCall2H
"dropout_31/StatefulPartitionedCall"dropout_31/StatefulPartitionedCall2H
"dropout_32/StatefulPartitionedCall"dropout_32/StatefulPartitionedCall2H
"dropout_33/StatefulPartitionedCall"dropout_33/StatefulPartitionedCall2H
"dropout_34/StatefulPartitionedCall"dropout_34/StatefulPartitionedCall2J
#features256/StatefulPartitionedCall#features256/StatefulPartitionedCall2J
#features512/StatefulPartitionedCall#features512/StatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿc(
 
_user_specified_nameinputs
	
Ñ
6__inference_batch_normalization_32_layer_call_fn_18064

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_batch_normalization_32_layer_call_and_return_conditional_losses_15238
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ø
c
E__inference_dropout_31_layer_call_and_return_conditional_losses_15983

inputs

identity_1V
IdentityIdentityinputs*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@c

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
	
Ñ
6__inference_batch_normalization_34_layer_call_fn_18260

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_batch_normalization_34_layer_call_and_return_conditional_losses_15347
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Â
F
*__inference_dropout_29_layer_call_fn_18334

inputs
identity»
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_29_layer_call_and_return_conditional_losses_15903h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ
:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs

g
K__inference_max_pooling2d_23_layer_call_and_return_conditional_losses_15550

inputs
identity¢
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ç
_
C__inference_re_lu_29_layer_call_and_return_conditional_losses_15895

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1b
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ1:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
 
_user_specified_nameinputs
Ë
_
C__inference_re_lu_33_layer_call_and_return_conditional_losses_16063

inputs
identityG
ReluReluinputs*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ì

Q__inference_batch_normalization_33_layer_call_and_return_conditional_losses_18163

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0È
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ°
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs"¿L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*¾
serving_defaultª
O
reshape_4_input<
!serving_default_reshape_4_input:0ÿÿÿÿÿÿÿÿÿc(;
dense_40
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:±Û
ù
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer-4
layer-5
layer-6
layer_with_weights-3
layer-7
	layer_with_weights-4
	layer-8

layer-9
layer-10
layer-11
layer_with_weights-5
layer-12
layer_with_weights-6
layer-13
layer-14
layer-15
layer-16
layer_with_weights-7
layer-17
layer_with_weights-8
layer-18
layer-19
layer-20
layer-21
layer_with_weights-9
layer-22
layer_with_weights-10
layer-23
layer-24
layer-25
layer-26
layer-27
layer_with_weights-11
layer-28
layer_with_weights-12
layer-29
layer-30
 layer-31
!layer_with_weights-13
!layer-32
"layer_with_weights-14
"layer-33
#layer-34
$layer-35
%layer_with_weights-15
%layer-36
&	variables
'trainable_variables
(regularization_losses
)	keras_api
*__call__
*+&call_and_return_all_conditional_losses
,_default_save_signature
-
signatures"
_tf_keras_sequential
¥
.	variables
/trainable_variables
0regularization_losses
1	keras_api
2__call__
*3&call_and_return_all_conditional_losses"
_tf_keras_layer
ê
4	variables
5trainable_variables
6regularization_losses
7	keras_api
8__call__
*9&call_and_return_all_conditional_losses
:axis
	;gamma
<beta
=moving_mean
>moving_variance"
_tf_keras_layer
Ý
?	variables
@trainable_variables
Aregularization_losses
B	keras_api
C__call__
*D&call_and_return_all_conditional_losses

Ekernel
Fbias
 G_jit_compiled_convolution_op"
_tf_keras_layer
ê
H	variables
Itrainable_variables
Jregularization_losses
K	keras_api
L__call__
*M&call_and_return_all_conditional_losses
Naxis
	Ogamma
Pbeta
Qmoving_mean
Rmoving_variance"
_tf_keras_layer
¥
S	variables
Ttrainable_variables
Uregularization_losses
V	keras_api
W__call__
*X&call_and_return_all_conditional_losses"
_tf_keras_layer
¥
Y	variables
Ztrainable_variables
[regularization_losses
\	keras_api
]__call__
*^&call_and_return_all_conditional_losses"
_tf_keras_layer
¼
_	variables
`trainable_variables
aregularization_losses
b	keras_api
c__call__
*d&call_and_return_all_conditional_losses
e_random_generator"
_tf_keras_layer
Ý
f	variables
gtrainable_variables
hregularization_losses
i	keras_api
j__call__
*k&call_and_return_all_conditional_losses

lkernel
mbias
 n_jit_compiled_convolution_op"
_tf_keras_layer
ê
o	variables
ptrainable_variables
qregularization_losses
r	keras_api
s__call__
*t&call_and_return_all_conditional_losses
uaxis
	vgamma
wbeta
xmoving_mean
ymoving_variance"
_tf_keras_layer
¥
z	variables
{trainable_variables
|regularization_losses
}	keras_api
~__call__
*&call_and_return_all_conditional_losses"
_tf_keras_layer
«
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
Ã
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses
_random_generator"
_tf_keras_layer
æ
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses
kernel
	bias
!_jit_compiled_convolution_op"
_tf_keras_layer
õ
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses
	axis

gamma
	beta
moving_mean
 moving_variance"
_tf_keras_layer
«
¡	variables
¢trainable_variables
£regularization_losses
¤	keras_api
¥__call__
+¦&call_and_return_all_conditional_losses"
_tf_keras_layer
«
§	variables
¨trainable_variables
©regularization_losses
ª	keras_api
«__call__
+¬&call_and_return_all_conditional_losses"
_tf_keras_layer
Ã
­	variables
®trainable_variables
¯regularization_losses
°	keras_api
±__call__
+²&call_and_return_all_conditional_losses
³_random_generator"
_tf_keras_layer
æ
´	variables
µtrainable_variables
¶regularization_losses
·	keras_api
¸__call__
+¹&call_and_return_all_conditional_losses
ºkernel
	»bias
!¼_jit_compiled_convolution_op"
_tf_keras_layer
õ
½	variables
¾trainable_variables
¿regularization_losses
À	keras_api
Á__call__
+Â&call_and_return_all_conditional_losses
	Ãaxis

Ägamma
	Åbeta
Æmoving_mean
Çmoving_variance"
_tf_keras_layer
«
È	variables
Étrainable_variables
Êregularization_losses
Ë	keras_api
Ì__call__
+Í&call_and_return_all_conditional_losses"
_tf_keras_layer
«
Î	variables
Ïtrainable_variables
Ðregularization_losses
Ñ	keras_api
Ò__call__
+Ó&call_and_return_all_conditional_losses"
_tf_keras_layer
Ã
Ô	variables
Õtrainable_variables
Öregularization_losses
×	keras_api
Ø__call__
+Ù&call_and_return_all_conditional_losses
Ú_random_generator"
_tf_keras_layer
æ
Û	variables
Ütrainable_variables
Ýregularization_losses
Þ	keras_api
ß__call__
+à&call_and_return_all_conditional_losses
ákernel
	âbias
!ã_jit_compiled_convolution_op"
_tf_keras_layer
õ
ä	variables
åtrainable_variables
æregularization_losses
ç	keras_api
è__call__
+é&call_and_return_all_conditional_losses
	êaxis

ëgamma
	ìbeta
ímoving_mean
îmoving_variance"
_tf_keras_layer
«
ï	variables
ðtrainable_variables
ñregularization_losses
ò	keras_api
ó__call__
+ô&call_and_return_all_conditional_losses"
_tf_keras_layer
«
õ	variables
ötrainable_variables
÷regularization_losses
ø	keras_api
ù__call__
+ú&call_and_return_all_conditional_losses"
_tf_keras_layer
Ã
û	variables
ütrainable_variables
ýregularization_losses
þ	keras_api
ÿ__call__
+&call_and_return_all_conditional_losses
_random_generator"
_tf_keras_layer
«
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
Ã
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses
kernel
	bias"
_tf_keras_layer
õ
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses
	axis

gamma
	beta
moving_mean
moving_variance"
_tf_keras_layer
«
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+ &call_and_return_all_conditional_losses"
_tf_keras_layer
Ã
¡	variables
¢trainable_variables
£regularization_losses
¤	keras_api
¥__call__
+¦&call_and_return_all_conditional_losses
§_random_generator"
_tf_keras_layer
Ã
¨	variables
©trainable_variables
ªregularization_losses
«	keras_api
¬__call__
+­&call_and_return_all_conditional_losses
®kernel
	¯bias"
_tf_keras_layer
õ
°	variables
±trainable_variables
²regularization_losses
³	keras_api
´__call__
+µ&call_and_return_all_conditional_losses
	¶axis

·gamma
	¸beta
¹moving_mean
ºmoving_variance"
_tf_keras_layer
«
»	variables
¼trainable_variables
½regularization_losses
¾	keras_api
¿__call__
+À&call_and_return_all_conditional_losses"
_tf_keras_layer
Ã
Á	variables
Âtrainable_variables
Ãregularization_losses
Ä	keras_api
Å__call__
+Æ&call_and_return_all_conditional_losses
Ç_random_generator"
_tf_keras_layer
Ã
È	variables
Étrainable_variables
Êregularization_losses
Ë	keras_api
Ì__call__
+Í&call_and_return_all_conditional_losses
Îkernel
	Ïbias"
_tf_keras_layer
¶
;0
<1
=2
>3
E4
F5
O6
P7
Q8
R9
l10
m11
v12
w13
x14
y15
16
17
18
19
20
 21
º22
»23
Ä24
Å25
Æ26
Ç27
á28
â29
ë30
ì31
í32
î33
34
35
36
37
38
39
®40
¯41
·42
¸43
¹44
º45
Î46
Ï47"
trackable_list_wrapper
¬
;0
<1
E2
F3
O4
P5
l6
m7
v8
w9
10
11
12
13
º14
»15
Ä16
Å17
á18
â19
ë20
ì21
22
23
24
25
®26
¯27
·28
¸29
Î30
Ï31"
trackable_list_wrapper
 "
trackable_list_wrapper
Ï
Ðnon_trainable_variables
Ñlayers
Òmetrics
 Ólayer_regularization_losses
Ôlayer_metrics
&	variables
'trainable_variables
(regularization_losses
*__call__
,_default_save_signature
*+&call_and_return_all_conditional_losses
&+"call_and_return_conditional_losses"
_generic_user_object
î
Õtrace_0
Ötrace_1
×trace_2
Øtrace_32û
,__inference_sequential_4_layer_call_fn_16228
,__inference_sequential_4_layer_call_fn_17441
,__inference_sequential_4_layer_call_fn_17542
,__inference_sequential_4_layer_call_fn_16963À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 zÕtrace_0zÖtrace_1z×trace_2zØtrace_3
Ú
Ùtrace_0
Útrace_1
Ûtrace_2
Ütrace_32ç
G__inference_sequential_4_layer_call_and_return_conditional_losses_17742
G__inference_sequential_4_layer_call_and_return_conditional_losses_18019
G__inference_sequential_4_layer_call_and_return_conditional_losses_17100
G__inference_sequential_4_layer_call_and_return_conditional_losses_17237À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 zÙtrace_0zÚtrace_1zÛtrace_2zÜtrace_3
ÓBÐ
 __inference__wrapped_model_15185reshape_4_input"
²
FullArgSpec
args 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
-
Ýserving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
²
Þnon_trainable_variables
ßlayers
àmetrics
 álayer_regularization_losses
âlayer_metrics
.	variables
/trainable_variables
0regularization_losses
2__call__
*3&call_and_return_all_conditional_losses
&3"call_and_return_conditional_losses"
_generic_user_object
ï
ãtrace_02Ð
)__inference_reshape_4_layer_call_fn_18024¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zãtrace_0

ätrace_02ë
D__inference_reshape_4_layer_call_and_return_conditional_losses_18038¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zätrace_0
<
;0
<1
=2
>3"
trackable_list_wrapper
.
;0
<1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
ånon_trainable_variables
ælayers
çmetrics
 èlayer_regularization_losses
élayer_metrics
4	variables
5trainable_variables
6regularization_losses
8__call__
*9&call_and_return_all_conditional_losses
&9"call_and_return_conditional_losses"
_generic_user_object
â
êtrace_0
ëtrace_12§
6__inference_batch_normalization_32_layer_call_fn_18051
6__inference_batch_normalization_32_layer_call_fn_18064´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 zêtrace_0zëtrace_1

ìtrace_0
ítrace_12Ý
Q__inference_batch_normalization_32_layer_call_and_return_conditional_losses_18082
Q__inference_batch_normalization_32_layer_call_and_return_conditional_losses_18100´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 zìtrace_0zítrace_1
 "
trackable_list_wrapper
*:(2batch_normalization_32/gamma
):'2batch_normalization_32/beta
2:0 (2"batch_normalization_32/moving_mean
6:4 (2&batch_normalization_32/moving_variance
.
E0
F1"
trackable_list_wrapper
.
E0
F1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
înon_trainable_variables
ïlayers
ðmetrics
 ñlayer_regularization_losses
òlayer_metrics
?	variables
@trainable_variables
Aregularization_losses
C__call__
*D&call_and_return_all_conditional_losses
&D"call_and_return_conditional_losses"
_generic_user_object
ï
ótrace_02Ð
)__inference_conv2d_20_layer_call_fn_18109¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zótrace_0

ôtrace_02ë
D__inference_conv2d_20_layer_call_and_return_conditional_losses_18119¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zôtrace_0
*:(2conv2d_20/kernel
:2conv2d_20/bias
´2±®
£²
FullArgSpec'
args
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 0
<
O0
P1
Q2
R3"
trackable_list_wrapper
.
O0
P1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
õnon_trainable_variables
ölayers
÷metrics
 ølayer_regularization_losses
ùlayer_metrics
H	variables
Itrainable_variables
Jregularization_losses
L__call__
*M&call_and_return_all_conditional_losses
&M"call_and_return_conditional_losses"
_generic_user_object
â
útrace_0
ûtrace_12§
6__inference_batch_normalization_33_layer_call_fn_18132
6__inference_batch_normalization_33_layer_call_fn_18145´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 zútrace_0zûtrace_1

ütrace_0
ýtrace_12Ý
Q__inference_batch_normalization_33_layer_call_and_return_conditional_losses_18163
Q__inference_batch_normalization_33_layer_call_and_return_conditional_losses_18181´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 zütrace_0zýtrace_1
 "
trackable_list_wrapper
*:(2batch_normalization_33/gamma
):'2batch_normalization_33/beta
2:0 (2"batch_normalization_33/moving_mean
6:4 (2&batch_normalization_33/moving_variance
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
²
þnon_trainable_variables
ÿlayers
metrics
 layer_regularization_losses
layer_metrics
S	variables
Ttrainable_variables
Uregularization_losses
W__call__
*X&call_and_return_all_conditional_losses
&X"call_and_return_conditional_losses"
_generic_user_object
î
trace_02Ï
(__inference_re_lu_28_layer_call_fn_18186¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 ztrace_0

trace_02ê
C__inference_re_lu_28_layer_call_and_return_conditional_losses_18191¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 ztrace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
²
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
Y	variables
Ztrainable_variables
[regularization_losses
]__call__
*^&call_and_return_all_conditional_losses
&^"call_and_return_conditional_losses"
_generic_user_object
ö
trace_02×
0__inference_max_pooling2d_20_layer_call_fn_18196¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 ztrace_0

trace_02ò
K__inference_max_pooling2d_20_layer_call_and_return_conditional_losses_18201¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 ztrace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
²
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
_	variables
`trainable_variables
aregularization_losses
c__call__
*d&call_and_return_all_conditional_losses
&d"call_and_return_conditional_losses"
_generic_user_object
Ê
trace_0
trace_12
*__inference_dropout_28_layer_call_fn_18206
*__inference_dropout_28_layer_call_fn_18211´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 ztrace_0ztrace_1

trace_0
trace_12Å
E__inference_dropout_28_layer_call_and_return_conditional_losses_18216
E__inference_dropout_28_layer_call_and_return_conditional_losses_18228´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 ztrace_0ztrace_1
"
_generic_user_object
.
l0
m1"
trackable_list_wrapper
.
l0
m1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
f	variables
gtrainable_variables
hregularization_losses
j__call__
*k&call_and_return_all_conditional_losses
&k"call_and_return_conditional_losses"
_generic_user_object
ï
trace_02Ð
)__inference_conv2d_21_layer_call_fn_18237¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 ztrace_0

trace_02ë
D__inference_conv2d_21_layer_call_and_return_conditional_losses_18247¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 ztrace_0
*:(2conv2d_21/kernel
:2conv2d_21/bias
´2±®
£²
FullArgSpec'
args
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 0
<
v0
w1
x2
y3"
trackable_list_wrapper
.
v0
w1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
non_trainable_variables
layers
metrics
 layer_regularization_losses
 layer_metrics
o	variables
ptrainable_variables
qregularization_losses
s__call__
*t&call_and_return_all_conditional_losses
&t"call_and_return_conditional_losses"
_generic_user_object
â
¡trace_0
¢trace_12§
6__inference_batch_normalization_34_layer_call_fn_18260
6__inference_batch_normalization_34_layer_call_fn_18273´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 z¡trace_0z¢trace_1

£trace_0
¤trace_12Ý
Q__inference_batch_normalization_34_layer_call_and_return_conditional_losses_18291
Q__inference_batch_normalization_34_layer_call_and_return_conditional_losses_18309´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 z£trace_0z¤trace_1
 "
trackable_list_wrapper
*:(2batch_normalization_34/gamma
):'2batch_normalization_34/beta
2:0 (2"batch_normalization_34/moving_mean
6:4 (2&batch_normalization_34/moving_variance
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
²
¥non_trainable_variables
¦layers
§metrics
 ¨layer_regularization_losses
©layer_metrics
z	variables
{trainable_variables
|regularization_losses
~__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
î
ªtrace_02Ï
(__inference_re_lu_29_layer_call_fn_18314¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zªtrace_0

«trace_02ê
C__inference_re_lu_29_layer_call_and_return_conditional_losses_18319¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 z«trace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
¬non_trainable_variables
­layers
®metrics
 ¯layer_regularization_losses
°layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
ö
±trace_02×
0__inference_max_pooling2d_21_layer_call_fn_18324¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 z±trace_0

²trace_02ò
K__inference_max_pooling2d_21_layer_call_and_return_conditional_losses_18329¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 z²trace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
³non_trainable_variables
´layers
µmetrics
 ¶layer_regularization_losses
·layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
Ê
¸trace_0
¹trace_12
*__inference_dropout_29_layer_call_fn_18334
*__inference_dropout_29_layer_call_fn_18339´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 z¸trace_0z¹trace_1

ºtrace_0
»trace_12Å
E__inference_dropout_29_layer_call_and_return_conditional_losses_18344
E__inference_dropout_29_layer_call_and_return_conditional_losses_18356´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 zºtrace_0z»trace_1
"
_generic_user_object
0
0
1"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
¼non_trainable_variables
½layers
¾metrics
 ¿layer_regularization_losses
Àlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
ï
Átrace_02Ð
)__inference_conv2d_22_layer_call_fn_18365¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zÁtrace_0

Âtrace_02ë
D__inference_conv2d_22_layer_call_and_return_conditional_losses_18375¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zÂtrace_0
*:( 2conv2d_22/kernel
: 2conv2d_22/bias
´2±®
£²
FullArgSpec'
args
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 0
@
0
1
2
 3"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
Ãnon_trainable_variables
Älayers
Åmetrics
 Ælayer_regularization_losses
Çlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
â
Ètrace_0
Étrace_12§
6__inference_batch_normalization_35_layer_call_fn_18388
6__inference_batch_normalization_35_layer_call_fn_18401´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 zÈtrace_0zÉtrace_1

Êtrace_0
Ëtrace_12Ý
Q__inference_batch_normalization_35_layer_call_and_return_conditional_losses_18419
Q__inference_batch_normalization_35_layer_call_and_return_conditional_losses_18437´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 zÊtrace_0zËtrace_1
 "
trackable_list_wrapper
*:( 2batch_normalization_35/gamma
):' 2batch_normalization_35/beta
2:0  (2"batch_normalization_35/moving_mean
6:4  (2&batch_normalization_35/moving_variance
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
Ìnon_trainable_variables
Ílayers
Îmetrics
 Ïlayer_regularization_losses
Ðlayer_metrics
¡	variables
¢trainable_variables
£regularization_losses
¥__call__
+¦&call_and_return_all_conditional_losses
'¦"call_and_return_conditional_losses"
_generic_user_object
î
Ñtrace_02Ï
(__inference_re_lu_30_layer_call_fn_18442¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zÑtrace_0

Òtrace_02ê
C__inference_re_lu_30_layer_call_and_return_conditional_losses_18447¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zÒtrace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
Ónon_trainable_variables
Ôlayers
Õmetrics
 Ölayer_regularization_losses
×layer_metrics
§	variables
¨trainable_variables
©regularization_losses
«__call__
+¬&call_and_return_all_conditional_losses
'¬"call_and_return_conditional_losses"
_generic_user_object
ö
Øtrace_02×
0__inference_max_pooling2d_22_layer_call_fn_18452¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zØtrace_0

Ùtrace_02ò
K__inference_max_pooling2d_22_layer_call_and_return_conditional_losses_18457¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zÙtrace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
Únon_trainable_variables
Ûlayers
Ümetrics
 Ýlayer_regularization_losses
Þlayer_metrics
­	variables
®trainable_variables
¯regularization_losses
±__call__
+²&call_and_return_all_conditional_losses
'²"call_and_return_conditional_losses"
_generic_user_object
Ê
ßtrace_0
àtrace_12
*__inference_dropout_30_layer_call_fn_18462
*__inference_dropout_30_layer_call_fn_18467´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 zßtrace_0zàtrace_1

átrace_0
âtrace_12Å
E__inference_dropout_30_layer_call_and_return_conditional_losses_18472
E__inference_dropout_30_layer_call_and_return_conditional_losses_18484´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 zátrace_0zâtrace_1
"
_generic_user_object
0
º0
»1"
trackable_list_wrapper
0
º0
»1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
ãnon_trainable_variables
älayers
åmetrics
 ælayer_regularization_losses
çlayer_metrics
´	variables
µtrainable_variables
¶regularization_losses
¸__call__
+¹&call_and_return_all_conditional_losses
'¹"call_and_return_conditional_losses"
_generic_user_object
ï
ètrace_02Ð
)__inference_conv2d_23_layer_call_fn_18493¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zètrace_0

étrace_02ë
D__inference_conv2d_23_layer_call_and_return_conditional_losses_18503¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zétrace_0
*:( @2conv2d_23/kernel
:@2conv2d_23/bias
´2±®
£²
FullArgSpec'
args
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 0
@
Ä0
Å1
Æ2
Ç3"
trackable_list_wrapper
0
Ä0
Å1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
ênon_trainable_variables
ëlayers
ìmetrics
 ílayer_regularization_losses
îlayer_metrics
½	variables
¾trainable_variables
¿regularization_losses
Á__call__
+Â&call_and_return_all_conditional_losses
'Â"call_and_return_conditional_losses"
_generic_user_object
â
ïtrace_0
ðtrace_12§
6__inference_batch_normalization_36_layer_call_fn_18516
6__inference_batch_normalization_36_layer_call_fn_18529´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 zïtrace_0zðtrace_1

ñtrace_0
òtrace_12Ý
Q__inference_batch_normalization_36_layer_call_and_return_conditional_losses_18547
Q__inference_batch_normalization_36_layer_call_and_return_conditional_losses_18565´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 zñtrace_0zòtrace_1
 "
trackable_list_wrapper
*:(@2batch_normalization_36/gamma
):'@2batch_normalization_36/beta
2:0@ (2"batch_normalization_36/moving_mean
6:4@ (2&batch_normalization_36/moving_variance
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
ónon_trainable_variables
ôlayers
õmetrics
 ölayer_regularization_losses
÷layer_metrics
È	variables
Étrainable_variables
Êregularization_losses
Ì__call__
+Í&call_and_return_all_conditional_losses
'Í"call_and_return_conditional_losses"
_generic_user_object
î
øtrace_02Ï
(__inference_re_lu_31_layer_call_fn_18570¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zøtrace_0

ùtrace_02ê
C__inference_re_lu_31_layer_call_and_return_conditional_losses_18575¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zùtrace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
únon_trainable_variables
ûlayers
ümetrics
 ýlayer_regularization_losses
þlayer_metrics
Î	variables
Ïtrainable_variables
Ðregularization_losses
Ò__call__
+Ó&call_and_return_all_conditional_losses
'Ó"call_and_return_conditional_losses"
_generic_user_object
ö
ÿtrace_02×
0__inference_max_pooling2d_23_layer_call_fn_18580¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zÿtrace_0

trace_02ò
K__inference_max_pooling2d_23_layer_call_and_return_conditional_losses_18585¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 ztrace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
Ô	variables
Õtrainable_variables
Öregularization_losses
Ø__call__
+Ù&call_and_return_all_conditional_losses
'Ù"call_and_return_conditional_losses"
_generic_user_object
Ê
trace_0
trace_12
*__inference_dropout_31_layer_call_fn_18590
*__inference_dropout_31_layer_call_fn_18595´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 ztrace_0ztrace_1

trace_0
trace_12Å
E__inference_dropout_31_layer_call_and_return_conditional_losses_18600
E__inference_dropout_31_layer_call_and_return_conditional_losses_18612´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 ztrace_0ztrace_1
"
_generic_user_object
0
á0
â1"
trackable_list_wrapper
0
á0
â1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
Û	variables
Ütrainable_variables
Ýregularization_losses
ß__call__
+à&call_and_return_all_conditional_losses
'à"call_and_return_conditional_losses"
_generic_user_object
ï
trace_02Ð
)__inference_conv2d_24_layer_call_fn_18621¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 ztrace_0

trace_02ë
D__inference_conv2d_24_layer_call_and_return_conditional_losses_18631¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 ztrace_0
+:)@2conv2d_24/kernel
:2conv2d_24/bias
´2±®
£²
FullArgSpec'
args
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 0
@
ë0
ì1
í2
î3"
trackable_list_wrapper
0
ë0
ì1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
ä	variables
åtrainable_variables
æregularization_losses
è__call__
+é&call_and_return_all_conditional_losses
'é"call_and_return_conditional_losses"
_generic_user_object
â
trace_0
trace_12§
6__inference_batch_normalization_37_layer_call_fn_18644
6__inference_batch_normalization_37_layer_call_fn_18657´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 ztrace_0ztrace_1

trace_0
trace_12Ý
Q__inference_batch_normalization_37_layer_call_and_return_conditional_losses_18675
Q__inference_batch_normalization_37_layer_call_and_return_conditional_losses_18693´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 ztrace_0ztrace_1
 "
trackable_list_wrapper
+:)2batch_normalization_37/gamma
*:(2batch_normalization_37/beta
3:1 (2"batch_normalization_37/moving_mean
7:5 (2&batch_normalization_37/moving_variance
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
ï	variables
ðtrainable_variables
ñregularization_losses
ó__call__
+ô&call_and_return_all_conditional_losses
'ô"call_and_return_conditional_losses"
_generic_user_object
î
trace_02Ï
(__inference_re_lu_32_layer_call_fn_18698¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 ztrace_0

 trace_02ê
C__inference_re_lu_32_layer_call_and_return_conditional_losses_18703¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 z trace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
¡non_trainable_variables
¢layers
£metrics
 ¤layer_regularization_losses
¥layer_metrics
õ	variables
ötrainable_variables
÷regularization_losses
ù__call__
+ú&call_and_return_all_conditional_losses
'ú"call_and_return_conditional_losses"
_generic_user_object
ö
¦trace_02×
0__inference_max_pooling2d_24_layer_call_fn_18708¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 z¦trace_0

§trace_02ò
K__inference_max_pooling2d_24_layer_call_and_return_conditional_losses_18713¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 z§trace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
¨non_trainable_variables
©layers
ªmetrics
 «layer_regularization_losses
¬layer_metrics
û	variables
ütrainable_variables
ýregularization_losses
ÿ__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
Ê
­trace_0
®trace_12
*__inference_dropout_32_layer_call_fn_18718
*__inference_dropout_32_layer_call_fn_18723´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 z­trace_0z®trace_1

¯trace_0
°trace_12Å
E__inference_dropout_32_layer_call_and_return_conditional_losses_18728
E__inference_dropout_32_layer_call_and_return_conditional_losses_18740´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 z¯trace_0z°trace_1
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
±non_trainable_variables
²layers
³metrics
 ´layer_regularization_losses
µlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
ï
¶trace_02Ð
)__inference_flatten_4_layer_call_fn_18745¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 z¶trace_0

·trace_02ë
D__inference_flatten_4_layer_call_and_return_conditional_losses_18751¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 z·trace_0
0
0
1"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
¸non_trainable_variables
¹layers
ºmetrics
 »layer_regularization_losses
¼layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
ñ
½trace_02Ò
+__inference_features512_layer_call_fn_18760¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 z½trace_0

¾trace_02í
F__inference_features512_layer_call_and_return_conditional_losses_18770¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 z¾trace_0
&:$
2features512/kernel
:2features512/bias
@
0
1
2
3"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
¿non_trainable_variables
Àlayers
Ámetrics
 Âlayer_regularization_losses
Ãlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
â
Ätrace_0
Åtrace_12§
6__inference_batch_normalization_38_layer_call_fn_18783
6__inference_batch_normalization_38_layer_call_fn_18796´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 zÄtrace_0zÅtrace_1

Ætrace_0
Çtrace_12Ý
Q__inference_batch_normalization_38_layer_call_and_return_conditional_losses_18816
Q__inference_batch_normalization_38_layer_call_and_return_conditional_losses_18850´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 zÆtrace_0zÇtrace_1
 "
trackable_list_wrapper
+:)2batch_normalization_38/gamma
*:(2batch_normalization_38/beta
3:1 (2"batch_normalization_38/moving_mean
7:5 (2&batch_normalization_38/moving_variance
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
Ènon_trainable_variables
Élayers
Êmetrics
 Ëlayer_regularization_losses
Ìlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+ &call_and_return_all_conditional_losses
' "call_and_return_conditional_losses"
_generic_user_object
î
Ítrace_02Ï
(__inference_re_lu_33_layer_call_fn_18855¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zÍtrace_0

Îtrace_02ê
C__inference_re_lu_33_layer_call_and_return_conditional_losses_18860¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zÎtrace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
Ïnon_trainable_variables
Ðlayers
Ñmetrics
 Òlayer_regularization_losses
Ólayer_metrics
¡	variables
¢trainable_variables
£regularization_losses
¥__call__
+¦&call_and_return_all_conditional_losses
'¦"call_and_return_conditional_losses"
_generic_user_object
Ê
Ôtrace_0
Õtrace_12
*__inference_dropout_33_layer_call_fn_18865
*__inference_dropout_33_layer_call_fn_18870´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 zÔtrace_0zÕtrace_1

Ötrace_0
×trace_12Å
E__inference_dropout_33_layer_call_and_return_conditional_losses_18875
E__inference_dropout_33_layer_call_and_return_conditional_losses_18887´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 zÖtrace_0z×trace_1
"
_generic_user_object
0
®0
¯1"
trackable_list_wrapper
0
®0
¯1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
Ønon_trainable_variables
Ùlayers
Úmetrics
 Ûlayer_regularization_losses
Ülayer_metrics
¨	variables
©trainable_variables
ªregularization_losses
¬__call__
+­&call_and_return_all_conditional_losses
'­"call_and_return_conditional_losses"
_generic_user_object
ñ
Ýtrace_02Ò
+__inference_features256_layer_call_fn_18896¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zÝtrace_0

Þtrace_02í
F__inference_features256_layer_call_and_return_conditional_losses_18906¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zÞtrace_0
&:$
2features256/kernel
:2features256/bias
@
·0
¸1
¹2
º3"
trackable_list_wrapper
0
·0
¸1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
ßnon_trainable_variables
àlayers
ámetrics
 âlayer_regularization_losses
ãlayer_metrics
°	variables
±trainable_variables
²regularization_losses
´__call__
+µ&call_and_return_all_conditional_losses
'µ"call_and_return_conditional_losses"
_generic_user_object
â
ätrace_0
åtrace_12§
6__inference_batch_normalization_39_layer_call_fn_18919
6__inference_batch_normalization_39_layer_call_fn_18932´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 zätrace_0zåtrace_1

ætrace_0
çtrace_12Ý
Q__inference_batch_normalization_39_layer_call_and_return_conditional_losses_18952
Q__inference_batch_normalization_39_layer_call_and_return_conditional_losses_18986´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 zætrace_0zçtrace_1
 "
trackable_list_wrapper
+:)2batch_normalization_39/gamma
*:(2batch_normalization_39/beta
3:1 (2"batch_normalization_39/moving_mean
7:5 (2&batch_normalization_39/moving_variance
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
ènon_trainable_variables
élayers
êmetrics
 ëlayer_regularization_losses
ìlayer_metrics
»	variables
¼trainable_variables
½regularization_losses
¿__call__
+À&call_and_return_all_conditional_losses
'À"call_and_return_conditional_losses"
_generic_user_object
î
ítrace_02Ï
(__inference_re_lu_34_layer_call_fn_18991¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zítrace_0

îtrace_02ê
C__inference_re_lu_34_layer_call_and_return_conditional_losses_18996¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zîtrace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
ïnon_trainable_variables
ðlayers
ñmetrics
 òlayer_regularization_losses
ólayer_metrics
Á	variables
Âtrainable_variables
Ãregularization_losses
Å__call__
+Æ&call_and_return_all_conditional_losses
'Æ"call_and_return_conditional_losses"
_generic_user_object
Ê
ôtrace_0
õtrace_12
*__inference_dropout_34_layer_call_fn_19001
*__inference_dropout_34_layer_call_fn_19006´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 zôtrace_0zõtrace_1

ötrace_0
÷trace_12Å
E__inference_dropout_34_layer_call_and_return_conditional_losses_19011
E__inference_dropout_34_layer_call_and_return_conditional_losses_19023´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 zötrace_0z÷trace_1
"
_generic_user_object
0
Î0
Ï1"
trackable_list_wrapper
0
Î0
Ï1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
ønon_trainable_variables
ùlayers
úmetrics
 ûlayer_regularization_losses
ülayer_metrics
È	variables
Étrainable_variables
Êregularization_losses
Ì__call__
+Í&call_and_return_all_conditional_losses
'Í"call_and_return_conditional_losses"
_generic_user_object
í
ýtrace_02Î
'__inference_dense_4_layer_call_fn_19032¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zýtrace_0

þtrace_02é
B__inference_dense_4_layer_call_and_return_conditional_losses_19043¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zþtrace_0
!:	2dense_4/kernel
:2dense_4/bias
 
=0
>1
Q2
R3
x4
y5
6
 7
Æ8
Ç9
í10
î11
12
13
¹14
º15"
trackable_list_wrapper
¾
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
 31
!32
"33
#34
$35
%36"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
B
,__inference_sequential_4_layer_call_fn_16228reshape_4_input"À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
þBû
,__inference_sequential_4_layer_call_fn_17441inputs"À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
þBû
,__inference_sequential_4_layer_call_fn_17542inputs"À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
B
,__inference_sequential_4_layer_call_fn_16963reshape_4_input"À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
B
G__inference_sequential_4_layer_call_and_return_conditional_losses_17742inputs"À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
B
G__inference_sequential_4_layer_call_and_return_conditional_losses_18019inputs"À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
¢B
G__inference_sequential_4_layer_call_and_return_conditional_losses_17100reshape_4_input"À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
¢B
G__inference_sequential_4_layer_call_and_return_conditional_losses_17237reshape_4_input"À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ÒBÏ
#__inference_signature_wrapper_17340reshape_4_input"
²
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
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
ÝBÚ
)__inference_reshape_4_layer_call_fn_18024inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
øBõ
D__inference_reshape_4_layer_call_and_return_conditional_losses_18038inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
.
=0
>1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
üBù
6__inference_batch_normalization_32_layer_call_fn_18051inputs"´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
üBù
6__inference_batch_normalization_32_layer_call_fn_18064inputs"´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
B
Q__inference_batch_normalization_32_layer_call_and_return_conditional_losses_18082inputs"´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
B
Q__inference_batch_normalization_32_layer_call_and_return_conditional_losses_18100inputs"´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
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
ÝBÚ
)__inference_conv2d_20_layer_call_fn_18109inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
øBõ
D__inference_conv2d_20_layer_call_and_return_conditional_losses_18119inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
.
Q0
R1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
üBù
6__inference_batch_normalization_33_layer_call_fn_18132inputs"´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
üBù
6__inference_batch_normalization_33_layer_call_fn_18145inputs"´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
B
Q__inference_batch_normalization_33_layer_call_and_return_conditional_losses_18163inputs"´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
B
Q__inference_batch_normalization_33_layer_call_and_return_conditional_losses_18181inputs"´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
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
ÜBÙ
(__inference_re_lu_28_layer_call_fn_18186inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
÷Bô
C__inference_re_lu_28_layer_call_and_return_conditional_losses_18191inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
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
äBá
0__inference_max_pooling2d_20_layer_call_fn_18196inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ÿBü
K__inference_max_pooling2d_20_layer_call_and_return_conditional_losses_18201inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
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
ðBí
*__inference_dropout_28_layer_call_fn_18206inputs"´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ðBí
*__inference_dropout_28_layer_call_fn_18211inputs"´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
B
E__inference_dropout_28_layer_call_and_return_conditional_losses_18216inputs"´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
B
E__inference_dropout_28_layer_call_and_return_conditional_losses_18228inputs"´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
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
ÝBÚ
)__inference_conv2d_21_layer_call_fn_18237inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
øBõ
D__inference_conv2d_21_layer_call_and_return_conditional_losses_18247inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
.
x0
y1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
üBù
6__inference_batch_normalization_34_layer_call_fn_18260inputs"´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
üBù
6__inference_batch_normalization_34_layer_call_fn_18273inputs"´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
B
Q__inference_batch_normalization_34_layer_call_and_return_conditional_losses_18291inputs"´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
B
Q__inference_batch_normalization_34_layer_call_and_return_conditional_losses_18309inputs"´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
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
ÜBÙ
(__inference_re_lu_29_layer_call_fn_18314inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
÷Bô
C__inference_re_lu_29_layer_call_and_return_conditional_losses_18319inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
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
äBá
0__inference_max_pooling2d_21_layer_call_fn_18324inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ÿBü
K__inference_max_pooling2d_21_layer_call_and_return_conditional_losses_18329inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
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
ðBí
*__inference_dropout_29_layer_call_fn_18334inputs"´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ðBí
*__inference_dropout_29_layer_call_fn_18339inputs"´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
B
E__inference_dropout_29_layer_call_and_return_conditional_losses_18344inputs"´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
B
E__inference_dropout_29_layer_call_and_return_conditional_losses_18356inputs"´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
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
ÝBÚ
)__inference_conv2d_22_layer_call_fn_18365inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
øBõ
D__inference_conv2d_22_layer_call_and_return_conditional_losses_18375inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
0
0
 1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
üBù
6__inference_batch_normalization_35_layer_call_fn_18388inputs"´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
üBù
6__inference_batch_normalization_35_layer_call_fn_18401inputs"´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
B
Q__inference_batch_normalization_35_layer_call_and_return_conditional_losses_18419inputs"´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
B
Q__inference_batch_normalization_35_layer_call_and_return_conditional_losses_18437inputs"´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
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
ÜBÙ
(__inference_re_lu_30_layer_call_fn_18442inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
÷Bô
C__inference_re_lu_30_layer_call_and_return_conditional_losses_18447inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
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
äBá
0__inference_max_pooling2d_22_layer_call_fn_18452inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ÿBü
K__inference_max_pooling2d_22_layer_call_and_return_conditional_losses_18457inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
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
ðBí
*__inference_dropout_30_layer_call_fn_18462inputs"´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ðBí
*__inference_dropout_30_layer_call_fn_18467inputs"´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
B
E__inference_dropout_30_layer_call_and_return_conditional_losses_18472inputs"´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
B
E__inference_dropout_30_layer_call_and_return_conditional_losses_18484inputs"´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
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
ÝBÚ
)__inference_conv2d_23_layer_call_fn_18493inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
øBõ
D__inference_conv2d_23_layer_call_and_return_conditional_losses_18503inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
0
Æ0
Ç1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
üBù
6__inference_batch_normalization_36_layer_call_fn_18516inputs"´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
üBù
6__inference_batch_normalization_36_layer_call_fn_18529inputs"´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
B
Q__inference_batch_normalization_36_layer_call_and_return_conditional_losses_18547inputs"´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
B
Q__inference_batch_normalization_36_layer_call_and_return_conditional_losses_18565inputs"´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
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
ÜBÙ
(__inference_re_lu_31_layer_call_fn_18570inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
÷Bô
C__inference_re_lu_31_layer_call_and_return_conditional_losses_18575inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
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
äBá
0__inference_max_pooling2d_23_layer_call_fn_18580inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ÿBü
K__inference_max_pooling2d_23_layer_call_and_return_conditional_losses_18585inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
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
ðBí
*__inference_dropout_31_layer_call_fn_18590inputs"´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ðBí
*__inference_dropout_31_layer_call_fn_18595inputs"´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
B
E__inference_dropout_31_layer_call_and_return_conditional_losses_18600inputs"´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
B
E__inference_dropout_31_layer_call_and_return_conditional_losses_18612inputs"´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
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
ÝBÚ
)__inference_conv2d_24_layer_call_fn_18621inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
øBõ
D__inference_conv2d_24_layer_call_and_return_conditional_losses_18631inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
0
í0
î1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
üBù
6__inference_batch_normalization_37_layer_call_fn_18644inputs"´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
üBù
6__inference_batch_normalization_37_layer_call_fn_18657inputs"´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
B
Q__inference_batch_normalization_37_layer_call_and_return_conditional_losses_18675inputs"´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
B
Q__inference_batch_normalization_37_layer_call_and_return_conditional_losses_18693inputs"´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
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
ÜBÙ
(__inference_re_lu_32_layer_call_fn_18698inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
÷Bô
C__inference_re_lu_32_layer_call_and_return_conditional_losses_18703inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
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
äBá
0__inference_max_pooling2d_24_layer_call_fn_18708inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ÿBü
K__inference_max_pooling2d_24_layer_call_and_return_conditional_losses_18713inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
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
ðBí
*__inference_dropout_32_layer_call_fn_18718inputs"´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ðBí
*__inference_dropout_32_layer_call_fn_18723inputs"´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
B
E__inference_dropout_32_layer_call_and_return_conditional_losses_18728inputs"´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
B
E__inference_dropout_32_layer_call_and_return_conditional_losses_18740inputs"´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
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
ÝBÚ
)__inference_flatten_4_layer_call_fn_18745inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
øBõ
D__inference_flatten_4_layer_call_and_return_conditional_losses_18751inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
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
ßBÜ
+__inference_features512_layer_call_fn_18760inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
úB÷
F__inference_features512_layer_call_and_return_conditional_losses_18770inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
üBù
6__inference_batch_normalization_38_layer_call_fn_18783inputs"´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
üBù
6__inference_batch_normalization_38_layer_call_fn_18796inputs"´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
B
Q__inference_batch_normalization_38_layer_call_and_return_conditional_losses_18816inputs"´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
B
Q__inference_batch_normalization_38_layer_call_and_return_conditional_losses_18850inputs"´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
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
ÜBÙ
(__inference_re_lu_33_layer_call_fn_18855inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
÷Bô
C__inference_re_lu_33_layer_call_and_return_conditional_losses_18860inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
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
ðBí
*__inference_dropout_33_layer_call_fn_18865inputs"´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ðBí
*__inference_dropout_33_layer_call_fn_18870inputs"´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
B
E__inference_dropout_33_layer_call_and_return_conditional_losses_18875inputs"´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
B
E__inference_dropout_33_layer_call_and_return_conditional_losses_18887inputs"´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
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
ßBÜ
+__inference_features256_layer_call_fn_18896inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
úB÷
F__inference_features256_layer_call_and_return_conditional_losses_18906inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
0
¹0
º1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
üBù
6__inference_batch_normalization_39_layer_call_fn_18919inputs"´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
üBù
6__inference_batch_normalization_39_layer_call_fn_18932inputs"´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
B
Q__inference_batch_normalization_39_layer_call_and_return_conditional_losses_18952inputs"´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
B
Q__inference_batch_normalization_39_layer_call_and_return_conditional_losses_18986inputs"´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
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
ÜBÙ
(__inference_re_lu_34_layer_call_fn_18991inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
÷Bô
C__inference_re_lu_34_layer_call_and_return_conditional_losses_18996inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
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
ðBí
*__inference_dropout_34_layer_call_fn_19001inputs"´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ðBí
*__inference_dropout_34_layer_call_fn_19006inputs"´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
B
E__inference_dropout_34_layer_call_and_return_conditional_losses_19011inputs"´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
B
E__inference_dropout_34_layer_call_and_return_conditional_losses_19023inputs"´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
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
ÛBØ
'__inference_dense_4_layer_call_fn_19032inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
öBó
B__inference_dense_4_layer_call_and_return_conditional_losses_19043inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 è
 __inference__wrapped_model_15185ÃP;<=>EFOPQRlmvwxy º»ÄÅÆÇáâëìíî®¯º·¹¸ÎÏ<¢9
2¢/
-*
reshape_4_inputÿÿÿÿÿÿÿÿÿc(
ª "1ª.
,
dense_4!
dense_4ÿÿÿÿÿÿÿÿÿì
Q__inference_batch_normalization_32_layer_call_and_return_conditional_losses_18082;<=>M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 ì
Q__inference_batch_normalization_32_layer_call_and_return_conditional_losses_18100;<=>M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Ä
6__inference_batch_normalization_32_layer_call_fn_18051;<=>M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÄ
6__inference_batch_normalization_32_layer_call_fn_18064;<=>M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿì
Q__inference_batch_normalization_33_layer_call_and_return_conditional_losses_18163OPQRM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 ì
Q__inference_batch_normalization_33_layer_call_and_return_conditional_losses_18181OPQRM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Ä
6__inference_batch_normalization_33_layer_call_fn_18132OPQRM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÄ
6__inference_batch_normalization_33_layer_call_fn_18145OPQRM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿì
Q__inference_batch_normalization_34_layer_call_and_return_conditional_losses_18291vwxyM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 ì
Q__inference_batch_normalization_34_layer_call_and_return_conditional_losses_18309vwxyM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Ä
6__inference_batch_normalization_34_layer_call_fn_18260vwxyM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÄ
6__inference_batch_normalization_34_layer_call_fn_18273vwxyM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿð
Q__inference_batch_normalization_35_layer_call_and_return_conditional_losses_18419 M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
p 
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 ð
Q__inference_batch_normalization_35_layer_call_and_return_conditional_losses_18437 M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
p
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 È
6__inference_batch_normalization_35_layer_call_fn_18388 M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
p 
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ È
6__inference_batch_normalization_35_layer_call_fn_18401 M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
p
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ ð
Q__inference_batch_normalization_36_layer_call_and_return_conditional_losses_18547ÄÅÆÇM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
p 
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 ð
Q__inference_batch_normalization_36_layer_call_and_return_conditional_losses_18565ÄÅÆÇM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
p
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 È
6__inference_batch_normalization_36_layer_call_fn_18516ÄÅÆÇM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
p 
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@È
6__inference_batch_normalization_36_layer_call_fn_18529ÄÅÆÇM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
p
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@ò
Q__inference_batch_normalization_37_layer_call_and_return_conditional_losses_18675ëìíîN¢K
D¢A
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª "@¢=
63
0,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 ò
Q__inference_batch_normalization_37_layer_call_and_return_conditional_losses_18693ëìíîN¢K
D¢A
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª "@¢=
63
0,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Ê
6__inference_batch_normalization_37_layer_call_fn_18644ëìíîN¢K
D¢A
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª "30,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÊ
6__inference_batch_normalization_37_layer_call_fn_18657ëìíîN¢K
D¢A
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª "30,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ½
Q__inference_batch_normalization_38_layer_call_and_return_conditional_losses_18816h4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 ½
Q__inference_batch_normalization_38_layer_call_and_return_conditional_losses_18850h4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ
p
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 
6__inference_batch_normalization_38_layer_call_fn_18783[4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "ÿÿÿÿÿÿÿÿÿ
6__inference_batch_normalization_38_layer_call_fn_18796[4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ
p
ª "ÿÿÿÿÿÿÿÿÿ½
Q__inference_batch_normalization_39_layer_call_and_return_conditional_losses_18952hº·¹¸4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 ½
Q__inference_batch_normalization_39_layer_call_and_return_conditional_losses_18986h¹º·¸4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ
p
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 
6__inference_batch_normalization_39_layer_call_fn_18919[º·¹¸4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "ÿÿÿÿÿÿÿÿÿ
6__inference_batch_normalization_39_layer_call_fn_18932[¹º·¸4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ
p
ª "ÿÿÿÿÿÿÿÿÿ´
D__inference_conv2d_20_layer_call_and_return_conditional_losses_18119lEF7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿc(
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿc(
 
)__inference_conv2d_20_layer_call_fn_18109_EF7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿc(
ª " ÿÿÿÿÿÿÿÿÿc(´
D__inference_conv2d_21_layer_call_and_return_conditional_losses_18247llm7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ1
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ1
 
)__inference_conv2d_21_layer_call_fn_18237_lm7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ1
ª " ÿÿÿÿÿÿÿÿÿ1¶
D__inference_conv2d_22_layer_call_and_return_conditional_losses_18375n7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ

ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ
 
 
)__inference_conv2d_22_layer_call_fn_18365a7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ

ª " ÿÿÿÿÿÿÿÿÿ
 ¶
D__inference_conv2d_23_layer_call_and_return_conditional_losses_18503nº»7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ@
 
)__inference_conv2d_23_layer_call_fn_18493aº»7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ 
ª " ÿÿÿÿÿÿÿÿÿ@·
D__inference_conv2d_24_layer_call_and_return_conditional_losses_18631oáâ7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ
 
)__inference_conv2d_24_layer_call_fn_18621báâ7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@
ª "!ÿÿÿÿÿÿÿÿÿ¥
B__inference_dense_4_layer_call_and_return_conditional_losses_19043_ÎÏ0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 }
'__inference_dense_4_layer_call_fn_19032RÎÏ0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿµ
E__inference_dropout_28_layer_call_and_return_conditional_losses_18216l;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ1
p 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ1
 µ
E__inference_dropout_28_layer_call_and_return_conditional_losses_18228l;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ1
p
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ1
 
*__inference_dropout_28_layer_call_fn_18206_;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ1
p 
ª " ÿÿÿÿÿÿÿÿÿ1
*__inference_dropout_28_layer_call_fn_18211_;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ1
p
ª " ÿÿÿÿÿÿÿÿÿ1µ
E__inference_dropout_29_layer_call_and_return_conditional_losses_18344l;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ

p 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ

 µ
E__inference_dropout_29_layer_call_and_return_conditional_losses_18356l;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ

p
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ

 
*__inference_dropout_29_layer_call_fn_18334_;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ

p 
ª " ÿÿÿÿÿÿÿÿÿ

*__inference_dropout_29_layer_call_fn_18339_;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ

p
ª " ÿÿÿÿÿÿÿÿÿ
µ
E__inference_dropout_30_layer_call_and_return_conditional_losses_18472l;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ 
p 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ 
 µ
E__inference_dropout_30_layer_call_and_return_conditional_losses_18484l;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ 
p
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ 
 
*__inference_dropout_30_layer_call_fn_18462_;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ 
p 
ª " ÿÿÿÿÿÿÿÿÿ 
*__inference_dropout_30_layer_call_fn_18467_;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ 
p
ª " ÿÿÿÿÿÿÿÿÿ µ
E__inference_dropout_31_layer_call_and_return_conditional_losses_18600l;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ@
p 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ@
 µ
E__inference_dropout_31_layer_call_and_return_conditional_losses_18612l;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ@
p
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ@
 
*__inference_dropout_31_layer_call_fn_18590_;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ@
p 
ª " ÿÿÿÿÿÿÿÿÿ@
*__inference_dropout_31_layer_call_fn_18595_;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ@
p
ª " ÿÿÿÿÿÿÿÿÿ@·
E__inference_dropout_32_layer_call_and_return_conditional_losses_18728n<¢9
2¢/
)&
inputsÿÿÿÿÿÿÿÿÿ
p 
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ
 ·
E__inference_dropout_32_layer_call_and_return_conditional_losses_18740n<¢9
2¢/
)&
inputsÿÿÿÿÿÿÿÿÿ
p
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ
 
*__inference_dropout_32_layer_call_fn_18718a<¢9
2¢/
)&
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "!ÿÿÿÿÿÿÿÿÿ
*__inference_dropout_32_layer_call_fn_18723a<¢9
2¢/
)&
inputsÿÿÿÿÿÿÿÿÿ
p
ª "!ÿÿÿÿÿÿÿÿÿ§
E__inference_dropout_33_layer_call_and_return_conditional_losses_18875^4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 §
E__inference_dropout_33_layer_call_and_return_conditional_losses_18887^4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ
p
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 
*__inference_dropout_33_layer_call_fn_18865Q4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "ÿÿÿÿÿÿÿÿÿ
*__inference_dropout_33_layer_call_fn_18870Q4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ
p
ª "ÿÿÿÿÿÿÿÿÿ§
E__inference_dropout_34_layer_call_and_return_conditional_losses_19011^4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 §
E__inference_dropout_34_layer_call_and_return_conditional_losses_19023^4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ
p
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 
*__inference_dropout_34_layer_call_fn_19001Q4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "ÿÿÿÿÿÿÿÿÿ
*__inference_dropout_34_layer_call_fn_19006Q4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ
p
ª "ÿÿÿÿÿÿÿÿÿª
F__inference_features256_layer_call_and_return_conditional_losses_18906`®¯0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 
+__inference_features256_layer_call_fn_18896S®¯0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿª
F__inference_features512_layer_call_and_return_conditional_losses_18770`0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 
+__inference_features512_layer_call_fn_18760S0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿª
D__inference_flatten_4_layer_call_and_return_conditional_losses_18751b8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 
)__inference_flatten_4_layer_call_fn_18745U8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿî
K__inference_max_pooling2d_20_layer_call_and_return_conditional_losses_18201R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Æ
0__inference_max_pooling2d_20_layer_call_fn_18196R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿî
K__inference_max_pooling2d_21_layer_call_and_return_conditional_losses_18329R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Æ
0__inference_max_pooling2d_21_layer_call_fn_18324R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿî
K__inference_max_pooling2d_22_layer_call_and_return_conditional_losses_18457R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Æ
0__inference_max_pooling2d_22_layer_call_fn_18452R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿî
K__inference_max_pooling2d_23_layer_call_and_return_conditional_losses_18585R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Æ
0__inference_max_pooling2d_23_layer_call_fn_18580R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿî
K__inference_max_pooling2d_24_layer_call_and_return_conditional_losses_18713R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Æ
0__inference_max_pooling2d_24_layer_call_fn_18708R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¯
C__inference_re_lu_28_layer_call_and_return_conditional_losses_18191h7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿc(
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿc(
 
(__inference_re_lu_28_layer_call_fn_18186[7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿc(
ª " ÿÿÿÿÿÿÿÿÿc(¯
C__inference_re_lu_29_layer_call_and_return_conditional_losses_18319h7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ1
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ1
 
(__inference_re_lu_29_layer_call_fn_18314[7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ1
ª " ÿÿÿÿÿÿÿÿÿ1¯
C__inference_re_lu_30_layer_call_and_return_conditional_losses_18447h7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ
 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ
 
 
(__inference_re_lu_30_layer_call_fn_18442[7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ
 
ª " ÿÿÿÿÿÿÿÿÿ
 ¯
C__inference_re_lu_31_layer_call_and_return_conditional_losses_18575h7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ@
 
(__inference_re_lu_31_layer_call_fn_18570[7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@
ª " ÿÿÿÿÿÿÿÿÿ@±
C__inference_re_lu_32_layer_call_and_return_conditional_losses_18703j8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ
 
(__inference_re_lu_32_layer_call_fn_18698]8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ
ª "!ÿÿÿÿÿÿÿÿÿ¡
C__inference_re_lu_33_layer_call_and_return_conditional_losses_18860Z0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 y
(__inference_re_lu_33_layer_call_fn_18855M0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ¡
C__inference_re_lu_34_layer_call_and_return_conditional_losses_18996Z0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 y
(__inference_re_lu_34_layer_call_fn_18991M0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ¬
D__inference_reshape_4_layer_call_and_return_conditional_losses_18038d3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿc(
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿc(
 
)__inference_reshape_4_layer_call_fn_18024W3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿc(
ª " ÿÿÿÿÿÿÿÿÿc(
G__inference_sequential_4_layer_call_and_return_conditional_losses_17100¿P;<=>EFOPQRlmvwxy º»ÄÅÆÇáâëìíî®¯º·¹¸ÎÏD¢A
:¢7
-*
reshape_4_inputÿÿÿÿÿÿÿÿÿc(
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
G__inference_sequential_4_layer_call_and_return_conditional_losses_17237¿P;<=>EFOPQRlmvwxy º»ÄÅÆÇáâëìíî®¯¹º·¸ÎÏD¢A
:¢7
-*
reshape_4_inputÿÿÿÿÿÿÿÿÿc(
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
G__inference_sequential_4_layer_call_and_return_conditional_losses_17742¶P;<=>EFOPQRlmvwxy º»ÄÅÆÇáâëìíî®¯º·¹¸ÎÏ;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿc(
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
G__inference_sequential_4_layer_call_and_return_conditional_losses_18019¶P;<=>EFOPQRlmvwxy º»ÄÅÆÇáâëìíî®¯¹º·¸ÎÏ;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿc(
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ã
,__inference_sequential_4_layer_call_fn_16228²P;<=>EFOPQRlmvwxy º»ÄÅÆÇáâëìíî®¯º·¹¸ÎÏD¢A
:¢7
-*
reshape_4_inputÿÿÿÿÿÿÿÿÿc(
p 

 
ª "ÿÿÿÿÿÿÿÿÿã
,__inference_sequential_4_layer_call_fn_16963²P;<=>EFOPQRlmvwxy º»ÄÅÆÇáâëìíî®¯¹º·¸ÎÏD¢A
:¢7
-*
reshape_4_inputÿÿÿÿÿÿÿÿÿc(
p

 
ª "ÿÿÿÿÿÿÿÿÿÚ
,__inference_sequential_4_layer_call_fn_17441©P;<=>EFOPQRlmvwxy º»ÄÅÆÇáâëìíî®¯º·¹¸ÎÏ;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿc(
p 

 
ª "ÿÿÿÿÿÿÿÿÿÚ
,__inference_sequential_4_layer_call_fn_17542©P;<=>EFOPQRlmvwxy º»ÄÅÆÇáâëìíî®¯¹º·¸ÎÏ;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿc(
p

 
ª "ÿÿÿÿÿÿÿÿÿþ
#__inference_signature_wrapper_17340ÖP;<=>EFOPQRlmvwxy º»ÄÅÆÇáâëìíî®¯º·¹¸ÎÏO¢L
¢ 
EªB
@
reshape_4_input-*
reshape_4_inputÿÿÿÿÿÿÿÿÿc("1ª.
,
dense_4!
dense_4ÿÿÿÿÿÿÿÿÿ