╓С
р╡
B
AssignVariableOp
resource
value"dtype"
dtypetypeИ
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
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(И
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
dtypetypeИ
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
list(type)(0И
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
?
Select
	condition

t"T
e"T
output"T"	
Ttype
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
9
Softmax
logits"T
softmax"T"
Ttype:
2
╛
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
executor_typestring И
@
StaticRegexFullMatch	
input

output
"
patternstring
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
Ц
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 И"serve*2.4.12v2.4.0-49-g85c8b2a817f8ў╙
z
dense_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
Ам*
shared_namedense_3/kernel
s
"dense_3/kernel/Read/ReadVariableOpReadVariableOpdense_3/kernel* 
_output_shapes
:
Ам*
dtype0
q
dense_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:м*
shared_namedense_3/bias
j
 dense_3/bias/Read/ReadVariableOpReadVariableOpdense_3/bias*
_output_shapes	
:м*
dtype0
z
dense_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
мЦ*
shared_namedense_4/kernel
s
"dense_4/kernel/Read/ReadVariableOpReadVariableOpdense_4/kernel* 
_output_shapes
:
мЦ*
dtype0
q
dense_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:Ц*
shared_namedense_4/bias
j
 dense_4/bias/Read/ReadVariableOpReadVariableOpdense_4/bias*
_output_shapes	
:Ц*
dtype0
y
dense_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	Ц*
shared_namedense_5/kernel
r
"dense_5/kernel/Read/ReadVariableOpReadVariableOpdense_5/kernel*
_output_shapes
:	Ц*
dtype0
p
dense_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_5/bias
i
 dense_5/bias/Read/ReadVariableOpReadVariableOpdense_5/bias*
_output_shapes
:*
dtype0

NoOpNoOp
╒
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*Р
valueЖBГ B№
Н
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer-5
	optimizer
regularization_losses
		variables

trainable_variables
	keras_api

signatures
R
regularization_losses
	variables
trainable_variables
	keras_api
h

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
h

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
R
regularization_losses
	variables
trainable_variables
 	keras_api
h

!kernel
"bias
#regularization_losses
$	variables
%trainable_variables
&	keras_api
R
'regularization_losses
(	variables
)trainable_variables
*	keras_api
 
 
*
0
1
2
3
!4
"5
*
0
1
2
3
!4
"5
н
+metrics
regularization_losses
		variables

,layers
-layer_regularization_losses

trainable_variables
.non_trainable_variables
/layer_metrics
 
 
 
 
н
0metrics
regularization_losses
	variables

1layers
2layer_regularization_losses
trainable_variables
3non_trainable_variables
4layer_metrics
ZX
VARIABLE_VALUEdense_3/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_3/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
н
5metrics
regularization_losses
	variables

6layers
7layer_regularization_losses
trainable_variables
8non_trainable_variables
9layer_metrics
ZX
VARIABLE_VALUEdense_4/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_4/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
н
:metrics
regularization_losses
	variables

;layers
<layer_regularization_losses
trainable_variables
=non_trainable_variables
>layer_metrics
 
 
 
н
?metrics
regularization_losses
	variables

@layers
Alayer_regularization_losses
trainable_variables
Bnon_trainable_variables
Clayer_metrics
ZX
VARIABLE_VALUEdense_5/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_5/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

!0
"1

!0
"1
н
Dmetrics
#regularization_losses
$	variables

Elayers
Flayer_regularization_losses
%trainable_variables
Gnon_trainable_variables
Hlayer_metrics
 
 
 
н
Imetrics
'regularization_losses
(	variables

Jlayers
Klayer_regularization_losses
)trainable_variables
Lnon_trainable_variables
Mlayer_metrics
 
*
0
1
2
3
4
5
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
Т
serving_default_flatten_1_inputPlaceholder*/
_output_shapes
:           *
dtype0*$
shape:           
Ю
StatefulPartitionedCallStatefulPartitionedCallserving_default_flatten_1_inputdense_3/kerneldense_3/biasdense_4/kerneldense_4/biasdense_5/kerneldense_5/bias*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8В *+
f&R$
"__inference_signature_wrapper_8141
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
ё
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename"dense_3/kernel/Read/ReadVariableOp dense_3/bias/Read/ReadVariableOp"dense_4/kernel/Read/ReadVariableOp dense_4/bias/Read/ReadVariableOp"dense_5/kernel/Read/ReadVariableOp dense_5/bias/Read/ReadVariableOpConst*
Tin

2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *&
f!R
__inference__traced_save_8359
Ї
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_3/kerneldense_3/biasdense_4/kerneldense_4/biasdense_5/kerneldense_5/bias*
Tin
	2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *)
f$R"
 __inference__traced_restore_8387ие
▄
╝
"__inference_signature_wrapper_8141
flatten_1_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identityИвStatefulPartitionedCallМ
StatefulPartitionedCallStatefulPartitionedCallflatten_1_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8В *(
f#R!
__inference__wrapped_model_78972
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:           ::::::22
StatefulPartitionedCallStatefulPartitionedCall:` \
/
_output_shapes
:           
)
_user_specified_nameflatten_1_input
╗
_
C__inference_flatten_1_layer_call_and_return_conditional_losses_7907

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:         А2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*.
_input_shapes
:           :W S
/
_output_shapes
:           
 
_user_specified_nameinputs
е
┐
E__inference_new_a_model_layer_call_and_return_conditional_losses_8021
flatten_1_input
dense_3_7937
dense_3_7939
dense_4_7963
dense_4_7965
dense_5_8002
dense_5_8004
identityИвdense_3/StatefulPartitionedCallвdense_4/StatefulPartitionedCallвdense_5/StatefulPartitionedCall▀
flatten_1/PartitionedCallPartitionedCallflatten_1_input*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_flatten_1_layer_call_and_return_conditional_losses_79072
flatten_1/PartitionedCallж
dense_3/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0dense_3_7937dense_3_7939*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         м*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_dense_3_layer_call_and_return_conditional_losses_79262!
dense_3/StatefulPartitionedCallм
dense_4/StatefulPartitionedCallStatefulPartitionedCall(dense_3/StatefulPartitionedCall:output:0dense_4_7963dense_4_7965*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         Ц*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_dense_4_layer_call_and_return_conditional_losses_79522!
dense_4/StatefulPartitionedCallБ
activation_2/PartitionedCallPartitionedCall(dense_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         Ц* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_activation_2_layer_call_and_return_conditional_losses_79732
activation_2/PartitionedCallи
dense_5/StatefulPartitionedCallStatefulPartitionedCall%activation_2/PartitionedCall:output:0dense_5_8002dense_5_8004*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_dense_5_layer_call_and_return_conditional_losses_79912!
dense_5/StatefulPartitionedCallА
activation_3/PartitionedCallPartitionedCall(dense_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_activation_3_layer_call_and_return_conditional_losses_80122
activation_3/PartitionedCall▀
IdentityIdentity%activation_3/PartitionedCall:output:0 ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:           ::::::2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall:` \
/
_output_shapes
:           
)
_user_specified_nameflatten_1_input
Т	
┌
A__inference_dense_5_layer_call_and_return_conditional_losses_8299

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	Ц*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2	
BiasAddХ
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*/
_input_shapes
:         Ц::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:         Ц
 
_user_specified_nameinputs
Ч	
┌
A__inference_dense_4_layer_call_and_return_conditional_losses_7952

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpП
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
мЦ*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ц2
MatMulН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:Ц*
dtype02
BiasAdd/ReadVariableOpВ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ц2	
BiasAddЦ
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:         Ц2

Identity"
identityIdentity:output:0*/
_input_shapes
:         м::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:         м
 
_user_specified_nameinputs
╝
b
F__inference_activation_3_layer_call_and_return_conditional_losses_8012

inputs
identityW
SoftmaxSoftmaxinputs*
T0*'
_output_shapes
:         2	
Softmaxe
IdentityIdentitySoftmax:softmax:0*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*&
_input_shapes
:         :O K
'
_output_shapes
:         
 
_user_specified_nameinputs
┘
■
__inference__traced_save_8359
file_prefix-
)savev2_dense_3_kernel_read_readvariableop+
'savev2_dense_3_bias_read_readvariableop-
)savev2_dense_4_kernel_read_readvariableop+
'savev2_dense_4_bias_read_readvariableop-
)savev2_dense_5_kernel_read_readvariableop+
'savev2_dense_5_bias_read_readvariableop
savev2_const

identity_1ИвMergeV2CheckpointsП
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
Const_1Л
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
ShardedFilename/shardж
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilenameы
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*¤
valueєBЁB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_namesЦ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*!
valueBB B B B B B B 2
SaveV2/shape_and_slices╝
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0)savev2_dense_3_kernel_read_readvariableop'savev2_dense_3_bias_read_readvariableop)savev2_dense_4_kernel_read_readvariableop'savev2_dense_4_bias_read_readvariableop)savev2_dense_5_kernel_read_readvariableop'savev2_dense_5_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
	22
SaveV2║
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixesб
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

identity_1Identity_1:output:0*N
_input_shapes=
;: :
Ам:м:
мЦ:Ц:	Ц:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:&"
 
_output_shapes
:
Ам:!

_output_shapes	
:м:&"
 
_output_shapes
:
мЦ:!

_output_shapes	
:Ц:%!

_output_shapes
:	Ц: 

_output_shapes
::

_output_shapes
: 
К
─
*__inference_new_a_model_layer_call_fn_8083
flatten_1_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identityИвStatefulPartitionedCall▓
StatefulPartitionedCallStatefulPartitionedCallflatten_1_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_new_a_model_layer_call_and_return_conditional_losses_80682
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:           ::::::22
StatefulPartitionedCallStatefulPartitionedCall:` \
/
_output_shapes
:           
)
_user_specified_nameflatten_1_input
ю
▓
E__inference_new_a_model_layer_call_and_return_conditional_losses_8168

inputs*
&dense_3_matmul_readvariableop_resource+
'dense_3_biasadd_readvariableop_resource*
&dense_4_matmul_readvariableop_resource+
'dense_4_biasadd_readvariableop_resource*
&dense_5_matmul_readvariableop_resource+
'dense_5_biasadd_readvariableop_resource
identityИвdense_3/BiasAdd/ReadVariableOpвdense_3/MatMul/ReadVariableOpвdense_4/BiasAdd/ReadVariableOpвdense_4/MatMul/ReadVariableOpвdense_5/BiasAdd/ReadVariableOpвdense_5/MatMul/ReadVariableOps
flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
flatten_1/ConstЖ
flatten_1/ReshapeReshapeinputsflatten_1/Const:output:0*
T0*(
_output_shapes
:         А2
flatten_1/Reshapeз
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource* 
_output_shapes
:
Ам*
dtype02
dense_3/MatMul/ReadVariableOpа
dense_3/MatMulMatMulflatten_1/Reshape:output:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         м2
dense_3/MatMulе
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes	
:м*
dtype02 
dense_3/BiasAdd/ReadVariableOpв
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         м2
dense_3/BiasAddz
dense_3/SigmoidSigmoiddense_3/BiasAdd:output:0*
T0*(
_output_shapes
:         м2
dense_3/Sigmoidз
dense_4/MatMul/ReadVariableOpReadVariableOp&dense_4_matmul_readvariableop_resource* 
_output_shapes
:
мЦ*
dtype02
dense_4/MatMul/ReadVariableOpЩ
dense_4/MatMulMatMuldense_3/Sigmoid:y:0%dense_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ц2
dense_4/MatMulе
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes	
:Ц*
dtype02 
dense_4/BiasAdd/ReadVariableOpв
dense_4/BiasAddBiasAdddense_4/MatMul:product:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ц2
dense_4/BiasAdd{
activation_2/TanhTanhdense_4/BiasAdd:output:0*
T0*(
_output_shapes
:         Ц2
activation_2/Tanhж
dense_5/MatMul/ReadVariableOpReadVariableOp&dense_5_matmul_readvariableop_resource*
_output_shapes
:	Ц*
dtype02
dense_5/MatMul/ReadVariableOpЪ
dense_5/MatMulMatMulactivation_2/Tanh:y:0%dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_5/MatMulд
dense_5/BiasAdd/ReadVariableOpReadVariableOp'dense_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_5/BiasAdd/ReadVariableOpб
dense_5/BiasAddBiasAdddense_5/MatMul:product:0&dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_5/BiasAddГ
activation_3/SoftmaxSoftmaxdense_5/BiasAdd:output:0*
T0*'
_output_shapes
:         2
activation_3/Softmax╡
IdentityIdentityactivation_3/Softmax:softmax:0^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:           ::::::2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp2@
dense_4/BiasAdd/ReadVariableOpdense_4/BiasAdd/ReadVariableOp2>
dense_4/MatMul/ReadVariableOpdense_4/MatMul/ReadVariableOp2@
dense_5/BiasAdd/ReadVariableOpdense_5/BiasAdd/ReadVariableOp2>
dense_5/MatMul/ReadVariableOpdense_5/MatMul/ReadVariableOp:W S
/
_output_shapes
:           
 
_user_specified_nameinputs
Ў	
┌
A__inference_dense_3_layer_call_and_return_conditional_losses_7926

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpП
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
Ам*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         м2
MatMulН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:м*
dtype02
BiasAdd/ReadVariableOpВ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         м2	
BiasAddb
SigmoidSigmoidBiasAdd:output:0*
T0*(
_output_shapes
:         м2	
SigmoidС
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:         м2

Identity"
identityIdentity:output:0*/
_input_shapes
:         А::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
К
─
*__inference_new_a_model_layer_call_fn_8122
flatten_1_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identityИвStatefulPartitionedCall▓
StatefulPartitionedCallStatefulPartitionedCallflatten_1_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_new_a_model_layer_call_and_return_conditional_losses_81072
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:           ::::::22
StatefulPartitionedCallStatefulPartitionedCall:` \
/
_output_shapes
:           
)
_user_specified_nameflatten_1_input
┌
{
&__inference_dense_3_layer_call_fn_8260

inputs
unknown
	unknown_0
identityИвStatefulPartitionedCallЄ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         м*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_dense_3_layer_call_and_return_conditional_losses_79262
StatefulPartitionedCallП
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:         м2

Identity"
identityIdentity:output:0*/
_input_shapes
:         А::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
·'
е
__inference__wrapped_model_7897
flatten_1_input6
2new_a_model_dense_3_matmul_readvariableop_resource7
3new_a_model_dense_3_biasadd_readvariableop_resource6
2new_a_model_dense_4_matmul_readvariableop_resource7
3new_a_model_dense_4_biasadd_readvariableop_resource6
2new_a_model_dense_5_matmul_readvariableop_resource7
3new_a_model_dense_5_biasadd_readvariableop_resource
identityИв*new_a_model/dense_3/BiasAdd/ReadVariableOpв)new_a_model/dense_3/MatMul/ReadVariableOpв*new_a_model/dense_4/BiasAdd/ReadVariableOpв)new_a_model/dense_4/MatMul/ReadVariableOpв*new_a_model/dense_5/BiasAdd/ReadVariableOpв)new_a_model/dense_5/MatMul/ReadVariableOpЛ
new_a_model/flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
new_a_model/flatten_1/Const│
new_a_model/flatten_1/ReshapeReshapeflatten_1_input$new_a_model/flatten_1/Const:output:0*
T0*(
_output_shapes
:         А2
new_a_model/flatten_1/Reshape╦
)new_a_model/dense_3/MatMul/ReadVariableOpReadVariableOp2new_a_model_dense_3_matmul_readvariableop_resource* 
_output_shapes
:
Ам*
dtype02+
)new_a_model/dense_3/MatMul/ReadVariableOp╨
new_a_model/dense_3/MatMulMatMul&new_a_model/flatten_1/Reshape:output:01new_a_model/dense_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         м2
new_a_model/dense_3/MatMul╔
*new_a_model/dense_3/BiasAdd/ReadVariableOpReadVariableOp3new_a_model_dense_3_biasadd_readvariableop_resource*
_output_shapes	
:м*
dtype02,
*new_a_model/dense_3/BiasAdd/ReadVariableOp╥
new_a_model/dense_3/BiasAddBiasAdd$new_a_model/dense_3/MatMul:product:02new_a_model/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         м2
new_a_model/dense_3/BiasAddЮ
new_a_model/dense_3/SigmoidSigmoid$new_a_model/dense_3/BiasAdd:output:0*
T0*(
_output_shapes
:         м2
new_a_model/dense_3/Sigmoid╦
)new_a_model/dense_4/MatMul/ReadVariableOpReadVariableOp2new_a_model_dense_4_matmul_readvariableop_resource* 
_output_shapes
:
мЦ*
dtype02+
)new_a_model/dense_4/MatMul/ReadVariableOp╔
new_a_model/dense_4/MatMulMatMulnew_a_model/dense_3/Sigmoid:y:01new_a_model/dense_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ц2
new_a_model/dense_4/MatMul╔
*new_a_model/dense_4/BiasAdd/ReadVariableOpReadVariableOp3new_a_model_dense_4_biasadd_readvariableop_resource*
_output_shapes	
:Ц*
dtype02,
*new_a_model/dense_4/BiasAdd/ReadVariableOp╥
new_a_model/dense_4/BiasAddBiasAdd$new_a_model/dense_4/MatMul:product:02new_a_model/dense_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ц2
new_a_model/dense_4/BiasAddЯ
new_a_model/activation_2/TanhTanh$new_a_model/dense_4/BiasAdd:output:0*
T0*(
_output_shapes
:         Ц2
new_a_model/activation_2/Tanh╩
)new_a_model/dense_5/MatMul/ReadVariableOpReadVariableOp2new_a_model_dense_5_matmul_readvariableop_resource*
_output_shapes
:	Ц*
dtype02+
)new_a_model/dense_5/MatMul/ReadVariableOp╩
new_a_model/dense_5/MatMulMatMul!new_a_model/activation_2/Tanh:y:01new_a_model/dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
new_a_model/dense_5/MatMul╚
*new_a_model/dense_5/BiasAdd/ReadVariableOpReadVariableOp3new_a_model_dense_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02,
*new_a_model/dense_5/BiasAdd/ReadVariableOp╤
new_a_model/dense_5/BiasAddBiasAdd$new_a_model/dense_5/MatMul:product:02new_a_model/dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
new_a_model/dense_5/BiasAddз
 new_a_model/activation_3/SoftmaxSoftmax$new_a_model/dense_5/BiasAdd:output:0*
T0*'
_output_shapes
:         2"
 new_a_model/activation_3/SoftmaxЙ
IdentityIdentity*new_a_model/activation_3/Softmax:softmax:0+^new_a_model/dense_3/BiasAdd/ReadVariableOp*^new_a_model/dense_3/MatMul/ReadVariableOp+^new_a_model/dense_4/BiasAdd/ReadVariableOp*^new_a_model/dense_4/MatMul/ReadVariableOp+^new_a_model/dense_5/BiasAdd/ReadVariableOp*^new_a_model/dense_5/MatMul/ReadVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:           ::::::2X
*new_a_model/dense_3/BiasAdd/ReadVariableOp*new_a_model/dense_3/BiasAdd/ReadVariableOp2V
)new_a_model/dense_3/MatMul/ReadVariableOp)new_a_model/dense_3/MatMul/ReadVariableOp2X
*new_a_model/dense_4/BiasAdd/ReadVariableOp*new_a_model/dense_4/BiasAdd/ReadVariableOp2V
)new_a_model/dense_4/MatMul/ReadVariableOp)new_a_model/dense_4/MatMul/ReadVariableOp2X
*new_a_model/dense_5/BiasAdd/ReadVariableOp*new_a_model/dense_5/BiasAdd/ReadVariableOp2V
)new_a_model/dense_5/MatMul/ReadVariableOp)new_a_model/dense_5/MatMul/ReadVariableOp:` \
/
_output_shapes
:           
)
_user_specified_nameflatten_1_input
Т	
┌
A__inference_dense_5_layer_call_and_return_conditional_losses_7991

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	Ц*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2	
BiasAddХ
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*/
_input_shapes
:         Ц::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:         Ц
 
_user_specified_nameinputs
╝
b
F__inference_activation_3_layer_call_and_return_conditional_losses_8313

inputs
identityW
SoftmaxSoftmaxinputs*
T0*'
_output_shapes
:         2	
Softmaxe
IdentityIdentitySoftmax:softmax:0*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*&
_input_shapes
:         :O K
'
_output_shapes
:         
 
_user_specified_nameinputs
о
b
F__inference_activation_2_layer_call_and_return_conditional_losses_7973

inputs
identityO
TanhTanhinputs*
T0*(
_output_shapes
:         Ц2
Tanh]
IdentityIdentityTanh:y:0*
T0*(
_output_shapes
:         Ц2

Identity"
identityIdentity:output:0*'
_input_shapes
:         Ц:P L
(
_output_shapes
:         Ц
 
_user_specified_nameinputs
К
╢
E__inference_new_a_model_layer_call_and_return_conditional_losses_8068

inputs
dense_3_8050
dense_3_8052
dense_4_8055
dense_4_8057
dense_5_8061
dense_5_8063
identityИвdense_3/StatefulPartitionedCallвdense_4/StatefulPartitionedCallвdense_5/StatefulPartitionedCall╓
flatten_1/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_flatten_1_layer_call_and_return_conditional_losses_79072
flatten_1/PartitionedCallж
dense_3/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0dense_3_8050dense_3_8052*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         м*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_dense_3_layer_call_and_return_conditional_losses_79262!
dense_3/StatefulPartitionedCallм
dense_4/StatefulPartitionedCallStatefulPartitionedCall(dense_3/StatefulPartitionedCall:output:0dense_4_8055dense_4_8057*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         Ц*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_dense_4_layer_call_and_return_conditional_losses_79522!
dense_4/StatefulPartitionedCallБ
activation_2/PartitionedCallPartitionedCall(dense_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         Ц* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_activation_2_layer_call_and_return_conditional_losses_79732
activation_2/PartitionedCallи
dense_5/StatefulPartitionedCallStatefulPartitionedCall%activation_2/PartitionedCall:output:0dense_5_8061dense_5_8063*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_dense_5_layer_call_and_return_conditional_losses_79912!
dense_5/StatefulPartitionedCallА
activation_3/PartitionedCallPartitionedCall(dense_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_activation_3_layer_call_and_return_conditional_losses_80122
activation_3/PartitionedCall▀
IdentityIdentity%activation_3/PartitionedCall:output:0 ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:           ::::::2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall:W S
/
_output_shapes
:           
 
_user_specified_nameinputs
о
b
F__inference_activation_2_layer_call_and_return_conditional_losses_8284

inputs
identityO
TanhTanhinputs*
T0*(
_output_shapes
:         Ц2
Tanh]
IdentityIdentityTanh:y:0*
T0*(
_output_shapes
:         Ц2

Identity"
identityIdentity:output:0*'
_input_shapes
:         Ц:P L
(
_output_shapes
:         Ц
 
_user_specified_nameinputs
Ч	
┌
A__inference_dense_4_layer_call_and_return_conditional_losses_8270

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpП
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
мЦ*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ц2
MatMulН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:Ц*
dtype02
BiasAdd/ReadVariableOpВ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ц2	
BiasAddЦ
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:         Ц2

Identity"
identityIdentity:output:0*/
_input_shapes
:         м::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:         м
 
_user_specified_nameinputs
ю
▓
E__inference_new_a_model_layer_call_and_return_conditional_losses_8195

inputs*
&dense_3_matmul_readvariableop_resource+
'dense_3_biasadd_readvariableop_resource*
&dense_4_matmul_readvariableop_resource+
'dense_4_biasadd_readvariableop_resource*
&dense_5_matmul_readvariableop_resource+
'dense_5_biasadd_readvariableop_resource
identityИвdense_3/BiasAdd/ReadVariableOpвdense_3/MatMul/ReadVariableOpвdense_4/BiasAdd/ReadVariableOpвdense_4/MatMul/ReadVariableOpвdense_5/BiasAdd/ReadVariableOpвdense_5/MatMul/ReadVariableOps
flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
flatten_1/ConstЖ
flatten_1/ReshapeReshapeinputsflatten_1/Const:output:0*
T0*(
_output_shapes
:         А2
flatten_1/Reshapeз
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource* 
_output_shapes
:
Ам*
dtype02
dense_3/MatMul/ReadVariableOpа
dense_3/MatMulMatMulflatten_1/Reshape:output:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         м2
dense_3/MatMulе
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes	
:м*
dtype02 
dense_3/BiasAdd/ReadVariableOpв
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         м2
dense_3/BiasAddz
dense_3/SigmoidSigmoiddense_3/BiasAdd:output:0*
T0*(
_output_shapes
:         м2
dense_3/Sigmoidз
dense_4/MatMul/ReadVariableOpReadVariableOp&dense_4_matmul_readvariableop_resource* 
_output_shapes
:
мЦ*
dtype02
dense_4/MatMul/ReadVariableOpЩ
dense_4/MatMulMatMuldense_3/Sigmoid:y:0%dense_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ц2
dense_4/MatMulе
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes	
:Ц*
dtype02 
dense_4/BiasAdd/ReadVariableOpв
dense_4/BiasAddBiasAdddense_4/MatMul:product:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ц2
dense_4/BiasAdd{
activation_2/TanhTanhdense_4/BiasAdd:output:0*
T0*(
_output_shapes
:         Ц2
activation_2/Tanhж
dense_5/MatMul/ReadVariableOpReadVariableOp&dense_5_matmul_readvariableop_resource*
_output_shapes
:	Ц*
dtype02
dense_5/MatMul/ReadVariableOpЪ
dense_5/MatMulMatMulactivation_2/Tanh:y:0%dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_5/MatMulд
dense_5/BiasAdd/ReadVariableOpReadVariableOp'dense_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_5/BiasAdd/ReadVariableOpб
dense_5/BiasAddBiasAdddense_5/MatMul:product:0&dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_5/BiasAddГ
activation_3/SoftmaxSoftmaxdense_5/BiasAdd:output:0*
T0*'
_output_shapes
:         2
activation_3/Softmax╡
IdentityIdentityactivation_3/Softmax:softmax:0^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:           ::::::2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp2@
dense_4/BiasAdd/ReadVariableOpdense_4/BiasAdd/ReadVariableOp2>
dense_4/MatMul/ReadVariableOpdense_4/MatMul/ReadVariableOp2@
dense_5/BiasAdd/ReadVariableOpdense_5/BiasAdd/ReadVariableOp2>
dense_5/MatMul/ReadVariableOpdense_5/MatMul/ReadVariableOp:W S
/
_output_shapes
:           
 
_user_specified_nameinputs
Ъ
G
+__inference_activation_2_layer_call_fn_8289

inputs
identity┼
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         Ц* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_activation_2_layer_call_and_return_conditional_losses_79732
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:         Ц2

Identity"
identityIdentity:output:0*'
_input_shapes
:         Ц:P L
(
_output_shapes
:         Ц
 
_user_specified_nameinputs
Ў	
┌
A__inference_dense_3_layer_call_and_return_conditional_losses_8251

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpП
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
Ам*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         м2
MatMulН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:м*
dtype02
BiasAdd/ReadVariableOpВ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         м2	
BiasAddb
SigmoidSigmoidBiasAdd:output:0*
T0*(
_output_shapes
:         м2	
SigmoidС
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:         м2

Identity"
identityIdentity:output:0*/
_input_shapes
:         А::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
я
╗
*__inference_new_a_model_layer_call_fn_8229

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identityИвStatefulPartitionedCallй
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_new_a_model_layer_call_and_return_conditional_losses_81072
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:           ::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:           
 
_user_specified_nameinputs
╗
_
C__inference_flatten_1_layer_call_and_return_conditional_losses_8235

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:         А2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*.
_input_shapes
:           :W S
/
_output_shapes
:           
 
_user_specified_nameinputs
╪
{
&__inference_dense_5_layer_call_fn_8308

inputs
unknown
	unknown_0
identityИвStatefulPartitionedCallё
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_dense_5_layer_call_and_return_conditional_losses_79912
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*/
_input_shapes
:         Ц::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         Ц
 
_user_specified_nameinputs
я
╗
*__inference_new_a_model_layer_call_fn_8212

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identityИвStatefulPartitionedCallй
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_new_a_model_layer_call_and_return_conditional_losses_80682
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:           ::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:           
 
_user_specified_nameinputs
Ц
G
+__inference_activation_3_layer_call_fn_8318

inputs
identity─
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_activation_3_layer_call_and_return_conditional_losses_80122
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*&
_input_shapes
:         :O K
'
_output_shapes
:         
 
_user_specified_nameinputs
З
д
 __inference__traced_restore_8387
file_prefix#
assignvariableop_dense_3_kernel#
assignvariableop_1_dense_3_bias%
!assignvariableop_2_dense_4_kernel#
assignvariableop_3_dense_4_bias%
!assignvariableop_4_dense_5_kernel#
assignvariableop_5_dense_5_bias

identity_7ИвAssignVariableOpвAssignVariableOp_1вAssignVariableOp_2вAssignVariableOp_3вAssignVariableOp_4вAssignVariableOp_5ё
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*¤
valueєBЁB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_namesЬ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*!
valueBB B B B B B B 2
RestoreV2/shape_and_slices╬
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*0
_output_shapes
:::::::*
dtypes
	22
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

IdentityЮ
AssignVariableOpAssignVariableOpassignvariableop_dense_3_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1д
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_3_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2ж
AssignVariableOp_2AssignVariableOp!assignvariableop_2_dense_4_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3д
AssignVariableOp_3AssignVariableOpassignvariableop_3_dense_4_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4ж
AssignVariableOp_4AssignVariableOp!assignvariableop_4_dense_5_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5д
AssignVariableOp_5AssignVariableOpassignvariableop_5_dense_5_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_59
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpф

Identity_6Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_6╓

Identity_7IdentityIdentity_6:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5*
T0*
_output_shapes
: 2

Identity_7"!

identity_7Identity_7:output:0*-
_input_shapes
: ::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_5:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
в
D
(__inference_flatten_1_layer_call_fn_8240

inputs
identity┬
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_flatten_1_layer_call_and_return_conditional_losses_79072
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*.
_input_shapes
:           :W S
/
_output_shapes
:           
 
_user_specified_nameinputs
К
╢
E__inference_new_a_model_layer_call_and_return_conditional_losses_8107

inputs
dense_3_8089
dense_3_8091
dense_4_8094
dense_4_8096
dense_5_8100
dense_5_8102
identityИвdense_3/StatefulPartitionedCallвdense_4/StatefulPartitionedCallвdense_5/StatefulPartitionedCall╓
flatten_1/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_flatten_1_layer_call_and_return_conditional_losses_79072
flatten_1/PartitionedCallж
dense_3/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0dense_3_8089dense_3_8091*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         м*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_dense_3_layer_call_and_return_conditional_losses_79262!
dense_3/StatefulPartitionedCallм
dense_4/StatefulPartitionedCallStatefulPartitionedCall(dense_3/StatefulPartitionedCall:output:0dense_4_8094dense_4_8096*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         Ц*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_dense_4_layer_call_and_return_conditional_losses_79522!
dense_4/StatefulPartitionedCallБ
activation_2/PartitionedCallPartitionedCall(dense_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         Ц* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_activation_2_layer_call_and_return_conditional_losses_79732
activation_2/PartitionedCallи
dense_5/StatefulPartitionedCallStatefulPartitionedCall%activation_2/PartitionedCall:output:0dense_5_8100dense_5_8102*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_dense_5_layer_call_and_return_conditional_losses_79912!
dense_5/StatefulPartitionedCallА
activation_3/PartitionedCallPartitionedCall(dense_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_activation_3_layer_call_and_return_conditional_losses_80122
activation_3/PartitionedCall▀
IdentityIdentity%activation_3/PartitionedCall:output:0 ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:           ::::::2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall:W S
/
_output_shapes
:           
 
_user_specified_nameinputs
е
┐
E__inference_new_a_model_layer_call_and_return_conditional_losses_8043
flatten_1_input
dense_3_8025
dense_3_8027
dense_4_8030
dense_4_8032
dense_5_8036
dense_5_8038
identityИвdense_3/StatefulPartitionedCallвdense_4/StatefulPartitionedCallвdense_5/StatefulPartitionedCall▀
flatten_1/PartitionedCallPartitionedCallflatten_1_input*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_flatten_1_layer_call_and_return_conditional_losses_79072
flatten_1/PartitionedCallж
dense_3/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0dense_3_8025dense_3_8027*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         м*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_dense_3_layer_call_and_return_conditional_losses_79262!
dense_3/StatefulPartitionedCallм
dense_4/StatefulPartitionedCallStatefulPartitionedCall(dense_3/StatefulPartitionedCall:output:0dense_4_8030dense_4_8032*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         Ц*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_dense_4_layer_call_and_return_conditional_losses_79522!
dense_4/StatefulPartitionedCallБ
activation_2/PartitionedCallPartitionedCall(dense_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         Ц* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_activation_2_layer_call_and_return_conditional_losses_79732
activation_2/PartitionedCallи
dense_5/StatefulPartitionedCallStatefulPartitionedCall%activation_2/PartitionedCall:output:0dense_5_8036dense_5_8038*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_dense_5_layer_call_and_return_conditional_losses_79912!
dense_5/StatefulPartitionedCallА
activation_3/PartitionedCallPartitionedCall(dense_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_activation_3_layer_call_and_return_conditional_losses_80122
activation_3/PartitionedCall▀
IdentityIdentity%activation_3/PartitionedCall:output:0 ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:           ::::::2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall:` \
/
_output_shapes
:           
)
_user_specified_nameflatten_1_input
┌
{
&__inference_dense_4_layer_call_fn_8279

inputs
unknown
	unknown_0
identityИвStatefulPartitionedCallЄ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         Ц*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_dense_4_layer_call_and_return_conditional_losses_79522
StatefulPartitionedCallП
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:         Ц2

Identity"
identityIdentity:output:0*/
_input_shapes
:         м::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         м
 
_user_specified_nameinputs"▒L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*╟
serving_default│
S
flatten_1_input@
!serving_default_flatten_1_input:0           @
activation_30
StatefulPartitionedCall:0         tensorflow/serving/predict:√л
╙(
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer-5
	optimizer
regularization_losses
		variables

trainable_variables
	keras_api

signatures
N__call__
O_default_save_signature
*P&call_and_return_all_conditional_losses"ь%
_tf_keras_sequential═%{"class_name": "Sequential", "name": "new_a_model", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "new_a_model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 32, 32, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "flatten_1_input"}}, {"class_name": "Flatten", "config": {"name": "flatten_1", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 32, 32, 1]}, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense_3", "trainable": true, "dtype": "float32", "units": 300, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_4", "trainable": true, "dtype": "float32", "units": 150, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_2", "trainable": true, "dtype": "float32", "activation": "tanh"}}, {"class_name": "Dense", "config": {"name": "dense_5", "trainable": true, "dtype": "float32", "units": 4, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_3", "trainable": true, "dtype": "float32", "activation": "softmax"}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32, 32, 1]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "new_a_model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 32, 32, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "flatten_1_input"}}, {"class_name": "Flatten", "config": {"name": "flatten_1", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 32, 32, 1]}, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense_3", "trainable": true, "dtype": "float32", "units": 300, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_4", "trainable": true, "dtype": "float32", "units": 150, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_2", "trainable": true, "dtype": "float32", "activation": "tanh"}}, {"class_name": "Dense", "config": {"name": "dense_5", "trainable": true, "dtype": "float32", "units": 4, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_3", "trainable": true, "dtype": "float32", "activation": "softmax"}}]}}, "training_config": {"loss": "categorical_crossentropy", "metrics": ["accuracy"], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 1e-05, "decay": 0, "beta_1": 0.9, "beta_2": 0.999, "epsilon": 1e-07, "amsgrad": false}}}}
ч
regularization_losses
	variables
trainable_variables
	keras_api
Q__call__
*R&call_and_return_all_conditional_losses"╪
_tf_keras_layer╛{"class_name": "Flatten", "name": "flatten_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 32, 32, 1]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten_1", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 32, 32, 1]}, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
є

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
S__call__
*T&call_and_return_all_conditional_losses"╬
_tf_keras_layer┤{"class_name": "Dense", "name": "dense_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_3", "trainable": true, "dtype": "float32", "units": 300, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 1024}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1024]}}
ї

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
U__call__
*V&call_and_return_all_conditional_losses"╨
_tf_keras_layer╢{"class_name": "Dense", "name": "dense_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_4", "trainable": true, "dtype": "float32", "units": 150, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 300}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 300]}}
╒
regularization_losses
	variables
trainable_variables
 	keras_api
W__call__
*X&call_and_return_all_conditional_losses"╞
_tf_keras_layerм{"class_name": "Activation", "name": "activation_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "activation_2", "trainable": true, "dtype": "float32", "activation": "tanh"}}
є

!kernel
"bias
#regularization_losses
$	variables
%trainable_variables
&	keras_api
Y__call__
*Z&call_and_return_all_conditional_losses"╬
_tf_keras_layer┤{"class_name": "Dense", "name": "dense_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_5", "trainable": true, "dtype": "float32", "units": 4, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 150}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 150]}}
╪
'regularization_losses
(	variables
)trainable_variables
*	keras_api
[__call__
*\&call_and_return_all_conditional_losses"╔
_tf_keras_layerп{"class_name": "Activation", "name": "activation_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "activation_3", "trainable": true, "dtype": "float32", "activation": "softmax"}}
"
	optimizer
 "
trackable_list_wrapper
J
0
1
2
3
!4
"5"
trackable_list_wrapper
J
0
1
2
3
!4
"5"
trackable_list_wrapper
╩
+metrics
regularization_losses
		variables

,layers
-layer_regularization_losses

trainable_variables
.non_trainable_variables
/layer_metrics
N__call__
O_default_save_signature
*P&call_and_return_all_conditional_losses
&P"call_and_return_conditional_losses"
_generic_user_object
,
]serving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
н
0metrics
regularization_losses
	variables

1layers
2layer_regularization_losses
trainable_variables
3non_trainable_variables
4layer_metrics
Q__call__
*R&call_and_return_all_conditional_losses
&R"call_and_return_conditional_losses"
_generic_user_object
": 
Ам2dense_3/kernel
:м2dense_3/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
н
5metrics
regularization_losses
	variables

6layers
7layer_regularization_losses
trainable_variables
8non_trainable_variables
9layer_metrics
S__call__
*T&call_and_return_all_conditional_losses
&T"call_and_return_conditional_losses"
_generic_user_object
": 
мЦ2dense_4/kernel
:Ц2dense_4/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
н
:metrics
regularization_losses
	variables

;layers
<layer_regularization_losses
trainable_variables
=non_trainable_variables
>layer_metrics
U__call__
*V&call_and_return_all_conditional_losses
&V"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
н
?metrics
regularization_losses
	variables

@layers
Alayer_regularization_losses
trainable_variables
Bnon_trainable_variables
Clayer_metrics
W__call__
*X&call_and_return_all_conditional_losses
&X"call_and_return_conditional_losses"
_generic_user_object
!:	Ц2dense_5/kernel
:2dense_5/bias
 "
trackable_list_wrapper
.
!0
"1"
trackable_list_wrapper
.
!0
"1"
trackable_list_wrapper
н
Dmetrics
#regularization_losses
$	variables

Elayers
Flayer_regularization_losses
%trainable_variables
Gnon_trainable_variables
Hlayer_metrics
Y__call__
*Z&call_and_return_all_conditional_losses
&Z"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
н
Imetrics
'regularization_losses
(	variables

Jlayers
Klayer_regularization_losses
)trainable_variables
Lnon_trainable_variables
Mlayer_metrics
[__call__
*\&call_and_return_all_conditional_losses
&\"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
J
0
1
2
3
4
5"
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
Ў2є
*__inference_new_a_model_layer_call_fn_8229
*__inference_new_a_model_layer_call_fn_8212
*__inference_new_a_model_layer_call_fn_8122
*__inference_new_a_model_layer_call_fn_8083└
╖▓│
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
э2ъ
__inference__wrapped_model_7897╞
Л▓З
FullArgSpec
argsЪ 
varargsjargs
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *6в3
1К.
flatten_1_input           
т2▀
E__inference_new_a_model_layer_call_and_return_conditional_losses_8195
E__inference_new_a_model_layer_call_and_return_conditional_losses_8043
E__inference_new_a_model_layer_call_and_return_conditional_losses_8021
E__inference_new_a_model_layer_call_and_return_conditional_losses_8168└
╖▓│
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
╥2╧
(__inference_flatten_1_layer_call_fn_8240в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
э2ъ
C__inference_flatten_1_layer_call_and_return_conditional_losses_8235в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╨2═
&__inference_dense_3_layer_call_fn_8260в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ы2ш
A__inference_dense_3_layer_call_and_return_conditional_losses_8251в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╨2═
&__inference_dense_4_layer_call_fn_8279в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ы2ш
A__inference_dense_4_layer_call_and_return_conditional_losses_8270в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╒2╥
+__inference_activation_2_layer_call_fn_8289в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
Ё2э
F__inference_activation_2_layer_call_and_return_conditional_losses_8284в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╨2═
&__inference_dense_5_layer_call_fn_8308в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ы2ш
A__inference_dense_5_layer_call_and_return_conditional_losses_8299в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╒2╥
+__inference_activation_3_layer_call_fn_8318в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
Ё2э
F__inference_activation_3_layer_call_and_return_conditional_losses_8313в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╤B╬
"__inference_signature_wrapper_8141flatten_1_input"Ф
Н▓Й
FullArgSpec
argsЪ 
varargs
 
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 л
__inference__wrapped_model_7897З!"@в=
6в3
1К.
flatten_1_input           
к ";к8
6
activation_3&К#
activation_3         д
F__inference_activation_2_layer_call_and_return_conditional_losses_8284Z0в-
&в#
!К
inputs         Ц
к "&в#
К
0         Ц
Ъ |
+__inference_activation_2_layer_call_fn_8289M0в-
&в#
!К
inputs         Ц
к "К         Цв
F__inference_activation_3_layer_call_and_return_conditional_losses_8313X/в,
%в"
 К
inputs         
к "%в"
К
0         
Ъ z
+__inference_activation_3_layer_call_fn_8318K/в,
%в"
 К
inputs         
к "К         г
A__inference_dense_3_layer_call_and_return_conditional_losses_8251^0в-
&в#
!К
inputs         А
к "&в#
К
0         м
Ъ {
&__inference_dense_3_layer_call_fn_8260Q0в-
&в#
!К
inputs         А
к "К         мг
A__inference_dense_4_layer_call_and_return_conditional_losses_8270^0в-
&в#
!К
inputs         м
к "&в#
К
0         Ц
Ъ {
&__inference_dense_4_layer_call_fn_8279Q0в-
&в#
!К
inputs         м
к "К         Цв
A__inference_dense_5_layer_call_and_return_conditional_losses_8299]!"0в-
&в#
!К
inputs         Ц
к "%в"
К
0         
Ъ z
&__inference_dense_5_layer_call_fn_8308P!"0в-
&в#
!К
inputs         Ц
к "К         и
C__inference_flatten_1_layer_call_and_return_conditional_losses_8235a7в4
-в*
(К%
inputs           
к "&в#
К
0         А
Ъ А
(__inference_flatten_1_layer_call_fn_8240T7в4
-в*
(К%
inputs           
к "К         А┬
E__inference_new_a_model_layer_call_and_return_conditional_losses_8021y!"HвE
>в;
1К.
flatten_1_input           
p

 
к "%в"
К
0         
Ъ ┬
E__inference_new_a_model_layer_call_and_return_conditional_losses_8043y!"HвE
>в;
1К.
flatten_1_input           
p 

 
к "%в"
К
0         
Ъ ╣
E__inference_new_a_model_layer_call_and_return_conditional_losses_8168p!"?в<
5в2
(К%
inputs           
p

 
к "%в"
К
0         
Ъ ╣
E__inference_new_a_model_layer_call_and_return_conditional_losses_8195p!"?в<
5в2
(К%
inputs           
p 

 
к "%в"
К
0         
Ъ Ъ
*__inference_new_a_model_layer_call_fn_8083l!"HвE
>в;
1К.
flatten_1_input           
p

 
к "К         Ъ
*__inference_new_a_model_layer_call_fn_8122l!"HвE
>в;
1К.
flatten_1_input           
p 

 
к "К         С
*__inference_new_a_model_layer_call_fn_8212c!"?в<
5в2
(К%
inputs           
p

 
к "К         С
*__inference_new_a_model_layer_call_fn_8229c!"?в<
5в2
(К%
inputs           
p 

 
к "К         ┴
"__inference_signature_wrapper_8141Ъ!"SвP
в 
IкF
D
flatten_1_input1К.
flatten_1_input           ";к8
6
activation_3&К#
activation_3         