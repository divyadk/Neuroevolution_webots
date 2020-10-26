/*
This is the version where based on the resource, the weights are
selected to do the mutation or not
Resource is either consumed or replenished based on the performance of the perturbed weights
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <webots/distance_sensor.h>
#include <webots/led.h>
#include <webots/light_sensor.h>
#include <webots/motor.h>
#include <webots/robot.h>

#define TIME_STEP 32 

// 8 IR proximity sensors of the epuck webots
#define NB_DIST_SENS 8
#define PS_RIGHT_00 0
#define PS_RIGHT_45 1
#define PS_RIGHT_90 2
#define PS_RIGHT_REAR 3
#define PS_LEFT_REAR 4
#define PS_LEFT_90 5
#define PS_LEFT_45 6
#define PS_LEFT_00 7

const int PS_OFFSET_SIMULATION[NB_DIST_SENS] = {300, 300, 300, 300, 300, 300, 300, 300};
const int PS_OFFSET_REALITY[NB_DIST_SENS] = {480, 170, 320, 500, 600, 680, 210, 640};
WbDeviceTag ps[NB_DIST_SENS]; /* proximity sensors */
int ps_value[NB_DIST_SENS] = {0, 0, 0, 0, 0, 0, 0, 0};
// Motors
WbDeviceTag left_motor, right_motor;
#define NB_LEDS 8
WbDeviceTag led[NB_LEDS];

//this is for the light sensor
WbDeviceTag ls;


//max_speed constant
#define MAX_SPEED 6.28

//these are the parameters for the forward propagation
#define SENSORS_N 9
#define HIDDEN_N 9
#define OUTPUT_N 2
#define SIGMA_MIN 0.01
#define SIGMA_MAX 4
#define PROB_REEVAL 0.3
#define RECOVERY 50
#define EVAL 100
#define INIT_FITNESS 10
#define SIGMA_CONSTRAINT 4
#define NO_OF_EVAL 600
#define SENSOR_MAX 150
#define SPEED_MAX 650
#define LIGHT_MAX 4200
#define REWARD 10
#define PENALTY 10
#define NO_SIGMA 10 //number of sigma values
#define MAX_RES 50  //max resource value
#define NO_MUT 20  //number of iterations to run in the beginning without guided mutation
//these are the global variables
double dist[NB_DIST_SENS];
double light;
int oeval; //this is to keep track of the number of eval in the controller
double weights1[HIDDEN_N][SENSORS_N];
double weights2[OUTPUT_N][HIDDEN_N];
double w1_sig_sti[HIDDEN_N*SENSORS_N][NO_SIGMA]; //w1 sigma stimulation
double w2_sig_sti[OUTPUT_N*HIDDEN_N][NO_SIGMA];  //w2 sigma stimulation
double output[OUTPUT_N][1];
double sigma=1.0;
double curr_ag[SENSORS_N];
double last_ag[SENSORS_N];
double last_fitness=INIT_FITNESS;
int ps_offset[NB_DIST_SENS] = {0, 0, 0, 0, 0, 0, 0, 0};

int mw1i[HIDDEN_N*SENSORS_N]; //mutate w1 index - indicates which all indices in w1 are mutated
int mw2i[OUTPUT_N*HIDDEN_N];  ////mutate w2 index - indicates which all indices in w2 are mutated
double w1_res[HIDDEN_N*SENSORS_N]; //weight matrix 1 - stimulation
double w1_res_copy[HIDDEN_N*SENSORS_N];  //copy of the matrix w1_res
double w2_res[OUTPUT_N*HIDDEN_N];  //weight matrix 2 - stimulation
double w2_res_copy[OUTPUT_N*HIDDEN_N];
int w1io[HIDDEN_N*SENSORS_N]; //this is the order of res indices ordered in increasing order
int w2io[OUTPUT_N*HIDDEN_N];
int nmut; //this is to indicate the number of mutations
//this is a structure for the weight
typedef struct champstruct{
  double cw1[HIDDEN_N][SENSORS_N];
  double cw2[OUTPUT_N][HIDDEN_N];
  double cf;
  int mw1i[HIDDEN_N*SENSORS_N]; //mutate w1 index - indicates which all indices in w1 are mutated
  int mw2i[OUTPUT_N*HIDDEN_N];  ////mutate w2 index - indicates which all indices in w2 are mutated
} Champ;

Champ champ,challenger;

//all other function definitions
//these are all the functions for the neuroevolution
void forward_prop(double w1[HIDDEN_N][SENSORS_N],double w2[OUTPUT_N][HIDDEN_N]);
void dot_all();
double sigmoid(double val);
void move_robot();
void init_champ();
void controller();
void recover();
double run_eval(double w1[HIDDEN_N][SENSORS_N],double w2[OUTPUT_N][HIDDEN_N]);
void copy_weights();
double cal_fitness();
void get_challenger();
void get_weights1();
void get_weights2();
double get_normv(double mu);
void update_champ();
void init_weights();
void get_inputs();
int get_max_dist();
//*************************************

FILE *wrlog;

void print_champ();
void print_challenger();
//main function starts from here
int main(void)
{
  int dummyvar=0;
  
  wb_robot_init(); //init the webots robot
  wrlog=fopen("test.txt","a");
  fprintf(wrlog, "This is the test log %d\n",dummyvar); //this is to write on the file
  printf("This is the test log %d\n",dummyvar);
  init_weights();
  init_champ(); 
  init_wres(); //this is to initialise the max_resource for the respective weight matrix
  //init_weight_sti();
  int i;
  //this is part of webots
  char name[20];
  for (i = 0; i < NB_LEDS; i++) {
    sprintf(name, "led%d", i);
    led[i] = wb_robot_get_device(name); /* get a handler to the sensor */
  }
  for (i = 0; i < NB_DIST_SENS; i++) {
    sprintf(name, "ps%d", i);
    ps[i] = wb_robot_get_device(name); /* proximity sensors */
    wb_distance_sensor_enable(ps[i], TIME_STEP);
  }

  //only one light sensor value the first one
  ls=wb_robot_get_device("ls0");
  wb_light_sensor_enable(ls,TIME_STEP);
  //light=wb_light_sensor_get_value(ls);
  //printf("Light sensor value is outside robot step%d\n",light);
  
  //setting motor values as zero
  // motors this is to initialise the values of hte robots to the custom values
  left_motor = wb_robot_get_device("left wheel motor");
  right_motor = wb_robot_get_device("right wheel motor");
  wb_motor_set_position(left_motor, INFINITY);
  wb_motor_set_position(right_motor, INFINITY);
  wb_motor_set_velocity(left_motor, 0.0);
  wb_motor_set_velocity(right_motor, 0.0);

  // read sensors value - this is unsure why is being done
    for (i = 0; i < NB_DIST_SENS; i++){
      ps_value[i] = (((int)wb_distance_sensor_get_value(ps[i]) - ps_offset[i] ) < 0) ?
                      0 :
                      ((int)wb_distance_sensor_get_value(ps[i]) - ps_offset[i] );
      dist[i]=ps_value[i];
    }
  // Speed initialization
    // speed[LEFT] = 0;
    // speed[RIGHT] = 0;
  for(;;){
    wb_robot_step(TIME_STEP);
    ls=wb_robot_get_device("ls0");
    wb_light_sensor_enable(ls,TIME_STEP);
    light=wb_light_sensor_get_value(ls);
    printf("Light sensor value is after robot step %lf\n",light);
    get_inputs();
    for(int i=0;i<NO_OF_EVAL;i++)
    {
      oeval=i;
      printf("*********************** EVALUATION %d***********************\n",i );
      fprintf(wrlog,"*********************** EVALUATION %d***********************\n",i );
      controller();
      printf("***********************\n");
      fprintf(wrlog,"***********************\n");
    }
    //here write to stop the bot
    wb_robot_cleanup(); //this will dismiss the controller
    fclose(wrlog);//end of wrlog
  }//end of infinite for loop
}


//this is to initialise the weights of the neural network
void init_weights(){
  fprintf(wrlog, "In init weights func and func testing\n");
  double r;
  for(int i=0;i<HIDDEN_N;i++)
  {
    for(int j=0;j<SENSORS_N;j++)
    { r=rand()/(double)RAND_MAX;
      weights1[i][j]= r;
    }
  }
  for(int i=0;i<OUTPUT_N;i++)
  {
    for(int j=0;j<HIDDEN_N;j++)
    { r=rand()/(double)RAND_MAX;
      weights2[i][j]= r;
    }
  }
}//end of function init_weights

//initilising the champ weights
void init_champ()
{
  copy_weights();
  champ.cf=-10; //initial fitness of the champ 
  for(int i=0;i>HIDDEN_N*SENSORS_N;i++)
    champ.mw1i[i]=0;
  for(int i=0;i>HIDDEN_N*OUTPUT_N;i++)
    champ.mw2i[i]=0;
}//end of function init_champ

//copying the weights of the neural network
void copy_weights()
{
  fprintf(wrlog, "init_champ [[");
  printf("init_champ [[");
  for(int i=0;i<HIDDEN_N;i++)
  {
    fprintf(wrlog, "[");
    printf("[");
    for(int j=0;j<SENSORS_N;j++)
    { 
      champ.cw1[i][j]= weights1[i][j];
      printf("%lf,",champ.cw1[i][j]);
      fprintf(wrlog,"%lf,",champ.cw1[i][j]);
    }
    fprintf(wrlog, "]");
    printf("]");
  }
  fprintf(wrlog, "],[");
  printf("],[");
  for(int i=0;i<OUTPUT_N;i++)
  {
    fprintf(wrlog, "[");
    printf("[");
    for(int j=0;j<HIDDEN_N;j++)
    { 
      champ.cw2[i][j]= weights2[i][j];
    }
    fprintf(wrlog, "]");
    printf("]");
  }
  fprintf(wrlog, "]]");
  printf("]]");
} //end of function copy_weights

//this is to get the inputs for the webots
void get_inputs(){
  int i;
// read sensors value
  wb_robot_step(TIME_STEP);
  //printf("initial_sensors [");
  //fprintf(wrlog,"initial_sensors [");
    for (i = 0; i < NB_DIST_SENS; i++){
     ps_value[i] = (((int)wb_distance_sensor_get_value(ps[i]) - ps_offset[i] ) < 0) ?
                      0 :
                      ((int)wb_distance_sensor_get_value(ps[i])- ps_offset[i]);
     if(ps_value[i]>149)
      ps_value[i]=150;
     dist[i]=(double)ps_value[i]/SENSOR_MAX;
     if(i==0){
      light=wb_light_sensor_get_value(ls);
      //printf("Light value is %lf\n",light );
      light=light/LIGHT_MAX;
     }
     //printf("distance %d %lf\n",ps_value[i],dist[i]);
     curr_ag[i]=dist[i];
     //printf("%lf,",curr_ag[i]);
     //fprintf(wrlog,"%lf,",curr_ag[i]);
   }
   //fprintf(wrlog, "Light val %lf\n",wb_light_sensor_get_value(ls) );
   light=wb_light_sensor_get_value(ls)/(double)LIGHT_MAX;
   curr_ag[i]=light;
   //curr_ag[i]=0; //this is for obstacle avoidance
   //printf("%lf]\n",curr_ag[i]);
   //fprintf(wrlog,"%lf]\n",curr_ag[i]);
}//end of the function get_inputs

//this is the main control function for the controller
void controller(){
  double rand_prob=rand()/(double)RAND_MAX;
  double prev_champf,cur_champf;
  if(rand_prob<PROB_REEVAL){    //re-evaluation of the champ
    printf("REEVALUATION OF THE CHAMP %lf\n",rand_prob);
    fprintf(wrlog,"REEVALUATION OF THE CHAMP %lf\n",rand_prob);
    prev_champf=champ.cf;
    printf("Fitness of the Champ before eval %lf\n",prev_champf);
    fprintf(wrlog,"Fitness of the Champ before eval %lf\n",prev_champf);
    recover(champ.cw1,champ.cw2);
    champ.cf=run_eval(champ.cw1,champ.cw2);
    printf("Champ Fitness after eval is %lf\n",champ.cf);
    fprintf(wrlog,"Champ Fitness after eval is %lf\n",champ.cf);
    //cur_champf=0.2*champ.cf+0.8*prev_champf;
    cur_champf=champ.cf;
    champ.cf=cur_champf;
  }//end of re evaluation if
  else
  {
    printf("CHALLENGER CASE %lf\n",rand_prob);
    fprintf(wrlog,"CHALLENGER CASE %lf\n",rand_prob);
    last_fitness=INIT_FITNESS;
    get_challenger();
    recover(challenger.cw1,challenger.cw2);
    challenger.cf=run_eval(challenger.cw1,challenger.cw2);
    //printf("Challenger fitness is %lf\n",challenger.cf);
    //fprintf(wrlog,"Challenger fitness is %lf\n",challenger.cf);
    if(challenger.cf>champ.cf+1)
    {
      printf("Challenger WINS! Chal %lf Champ %lf\n",challenger.cf,champ.cf );
      fprintf(wrlog,"Challenger WINS! Chal %lf Champ %lf\n",challenger.cf,champ.cf );
      update_champ();
      sigma=SIGMA_MIN;
      printf("Sigma updated value min is %lf\n",sigma);
      fprintf(wrlog,"Sigma updated value min is %lf\n",sigma);
    }//if challenger better than champ
    else
    {
      printf("Challenger LOSES! Chal %lf Champ %lf\n",challenger.cf,champ.cf );
      fprintf(wrlog,"Challenger LOSES! Chal %lf Champ %lf\n",challenger.cf,champ.cf );
      double tempsigma=sigma*2;
      if(tempsigma>=SIGMA_MAX)
        sigma=(double)SIGMA_MAX;
      else
        sigma=tempsigma;
      //printf("tempsigma value is %lf ; sigma max is %d\n",tempsigma,SIGMA_MAX);
      //fprintf(wrlog,"Sigma updated value is %lf\n",sigma);
      printf("Sigma updated value is %lf\n",sigma);
      fprintf(wrlog,"Sigma updated value is %lf\n",sigma);
    }//if challenger not better than champ
  }//end of challenger 
}//end of function controller

//this is to recover the initial few iterations
void recover(double w1[HIDDEN_N][SENSORS_N],double w2[OUTPUT_N][HIDDEN_N])
{
  for(int i=0;i<RECOVERY;i++)
  {
    get_inputs(); //this is to be MODIFIED
    forward_prop(w1,w2);
    move_robot(); //THIS IS TO BE MODIFIED
  }
}//end of recover function

void forward_prop(double w1[HIDDEN_N][SENSORS_N],double w2[OUTPUT_N][HIDDEN_N]){
  get_inputs();
  dot_all(w1,w2);
  move_robot();
}//end of the function forward_prop

//this is called from the forward_prop procedure
void dot_all(double w1[HIDDEN_N][SENSORS_N],double w2[OUTPUT_N][HIDDEN_N])
{
  double interP[HIDDEN_N][1];
  
  for(int i=0;i<HIDDEN_N;i++)
  { interP[i][0]=0;
    for(int j=0;j<SENSORS_N;j++)
    {
      interP[i][0]+=w1[i][j]*curr_ag[j];
    }
  }
  
  for(int i=0;i<HIDDEN_N;i++)
    interP[i][0]=sigmoid(interP[i][0]);
  
  for(int i=0;i<OUTPUT_N;i++)
  { output[i][0]=0;
    for(int j=0;j<HIDDEN_N;j++)
    {
      output[i][0]+=w2[i][j]*interP[j][0];
    }
  }
  
  output[0][0]=tanh(output[0][0]);
  output[1][0]=tanh(output[1][0]);
  
}//end of function dotall

//this is called from dot_all procedure - sigmoid function
double sigmoid(double x){
  float exp_value;
  float return_value;
  exp_value = exp((double) -x);
  return_value = 1 / (1 + exp_value);
  return return_value;
}

double run_eval(double w1[HIDDEN_N][SENSORS_N],double w2[OUTPUT_N][HIDDEN_N])
{
  double cur_fitness,prev_dist,cur_dist;
  last_fitness=INIT_FITNESS;
  get_inputs();
  prev_dist=(double)get_max_dist();
  for(int i=0;i<EVAL;i++)
  {
    get_inputs(); //this is to be MODIFIED
    forward_prop(w1,w2);
    move_robot();
    cur_fitness=cal_fitness();
    last_fitness+=cur_fitness;
  }

  //this is actually the post run_eval_meta in run_eval in the prolog version
  get_inputs();
  cur_dist=(double)get_max_dist();

  if(prev_dist>=150 && cur_dist<75){ //this means the obstacle is avoided
    last_fitness+=REWARD;
    fprintf(wrlog,"obsavoid Obstacle is avoided pdist %lf cdist %lf\n",prev_dist,cur_dist); 
    printf("obsavoid Obstacle is avoided pdist %lf cdist %lf\n",prev_dist,cur_dist); 
  }
  else if(prev_dist>=150 && cur_dist>=150){//obstacle not avoided give a penalty
    last_fitness=last_fitness-PENALTY;
    fprintf(wrlog,"obsnotavoid Obstacle NOT avoided pdist %lf cdist %lf\n",prev_dist,cur_dist); 
    printf("obsnotavoid Obstacle NOT avoided pdist %lf cdist %lf\n",prev_dist,cur_dist); 
  }
  else{
    //last_fitness=last_fitness-PENALTY;
    fprintf(wrlog,"normal case prev_dist %lf cdist %lf\n",prev_dist,cur_dist); 
    printf("normal case pdist %lf cdist %lf\n",prev_dist,cur_dist); 
  }
  fprintf(wrlog, "Overall fitness is %lf\n",last_fitness);
  printf("Overall Fitness %lf\n",last_fitness);
  return last_fitness;
}//end of run eval function

void move_robot()
{
  int m1,m2; //dir 0x06-> straight 0x05->left back,right forward 3->left forward,right back 4->both back
  m1=ceil(output[0][0]*SPEED_MAX); //taking the value as 650 that is the max speed ((CHANGE))
  m2=ceil(output[1][0]*SPEED_MAX);
  
  //below is to set the velocity of both the motors
  wb_motor_set_velocity(left_motor, 0.00628 * m1);
  wb_motor_set_velocity(right_motor, 0.00628 * m2);
}//end of move_robot function

//this is called from various places to get the max distance from the dist array
int get_max_dist(){
  int i;
  double max_dist=ps_value[0];
  for(i=0;i<NB_DIST_SENS;i++){
    if(max_dist<ps_value[i])
      max_dist=ps_value[i];
  }
  return max_dist;
}//end of the function get_max_dist

double cal_fitness(){
  double max_dist,flight,fitnesso,fitness;
  double trans_speed=(output[0][0]+output[1][0])/2;
  double rot_speed=(output[0][0]-output[1][0])/2;
  
  max_dist=(double)get_max_dist()/(double)SENSOR_MAX;
  flight=1 - light;
  fitnesso=trans_speed*rot_speed*(1-max_dist);
  fitness=flight+fitnesso;
  //fitness=fitnesso;
  return fitness;
}
//end of function cal_fitness


void get_challenger()
{
  //double w1[HIDDEN_N][SENSORS_N],w2[OUTPUT_N][HIDDEN_N];
  get_weights1();
  //get_weights1r();
  get_weights2();
  //get_weights2r();
  printf("Challenger (modified champ) after\n");
  fprintf(wrlog, "Challenger (modified champ) after\n");
  print_challenger();//this is to print the champ
}//end of get_challenger function

void get_weights1()
{
  double temp;
  double normv;
  int constraint=SIGMA_CONSTRAINT*-1;
  for(int i=0;i<HIDDEN_N;i++)
  {
    for(int j=0;j<SENSORS_N;j++)
    {
      normv=get_normv(0.0);
      temp=champ.cw1[i][j]+normv;
      if(temp>SIGMA_CONSTRAINT)
        challenger.cw1[i][j]=SIGMA_CONSTRAINT;
      else if(temp<constraint)
        challenger.cw1[i][j]=constraint;
      else
        challenger.cw1[i][j]=temp;
    }
  } 
}

void get_weights2()
{
  double temp;
  double normv;
  int constraint=SIGMA_CONSTRAINT*-1;
  for(int i=0;i<OUTPUT_N;i++)
  {
    for(int j=0;j<HIDDEN_N;j++)
    {
      normv=get_normv(0.0);
      temp=champ.cw2[i][j]+normv;
      if(temp>SIGMA_CONSTRAINT)
      challenger.cw2[i][j]=SIGMA_CONSTRAINT;
      else if(temp<constraint)
      challenger.cw2[i][j]=constraint;
      else
      challenger.cw2[i][j]=temp;
    }
  } 
}//END OF GET_WEIGHTS2

void update_champ()
{
  for(int i=0;i<HIDDEN_N;i++)
  {
    for(int j=0;j<SENSORS_N;j++)
    {
      champ.cw1[i][j]= challenger.cw1[i][j];
    }
  }
  for(int i=0;i<OUTPUT_N;i++)
  {
    for(int j=0;j<HIDDEN_N;j++)
    {
      champ.cw2[i][j]= challenger.cw2[i][j];
    }
  }
  champ.cf=challenger.cf;
  for(int i=0;i<SENSORS_N*HIDDEN_N;i++)
    champ.mw1i[i]=mw1i[i];
  for(int i=0;i<HIDDEN_N*OUTPUT_N;i++)
    champ.mw2i[i]=mw2i[i];
}//end of update_champ()

double get_normv(double mu)
{
  double r1=rand()/(double)RAND_MAX;
  double r2=rand()/(double)RAND_MAX;
  double normrand=mu+sigma*sqrt(-2*log(r1))*sin(6.28319*r2);
  return normrand;
} //getting a norm value end of function get_normv

void print_champ(){
  printf("[[");
  fprintf(wrlog, "[[");
  for(int i=0;i<HIDDEN_N;i++){
    for(int j=0;j<SENSORS_N;j++){
      printf("%lf,",champ.cw1[i][j] );
      fprintf(wrlog,"%lf,",champ.cw1[i][j] );
    }
  }
  printf("][");
  fprintf(wrlog, "][");
  for(int i=0;i<OUTPUT_N;i++){
    for(int j=0;j<HIDDEN_N;j++){
      printf("%lf,",champ.cw2[i][j] );
      fprintf(wrlog,"%lf,",champ.cw2[i][j] );
    }
  }
  printf("]]\n");
  fprintf(wrlog, "]]\n");
  printf("Champ Fitness %lf\n",champ.cf);
  fprintf(wrlog, "Champ Fitness %lf\n",champ.cf);
  printf("Mutation Indices Champ W1 [");
  fprintf(wrlog,"Mutation Indices Champ W1 [");
  for(int i=0;i<HIDDEN_N*SENSORS_N;i++){
    printf("%d,",champ.mw1i[i]);
    fprintf(wrlog, "%d,",champ.mw1i[i]);
  }//end of for loop
  printf("]\n");
  fprintf(wrlog,"]\n");
  printf("Mutation Indices Champ W2 [");
  fprintf(wrlog,"Mutation Indices Champ W2 [");
  for(int i=0;i<HIDDEN_N*OUTPUT_N;i++){
    printf("%d,",champ.mw2i[i]);
    fprintf(wrlog, "%d,",champ.mw2i[i]);
  }//end of for loop
  printf("]\n");
  fprintf(wrlog,"]\n");
}//end of print champ

void print_challenger(){
  printf("[[");
  fprintf(wrlog, "[[");
  for(int i=0;i<HIDDEN_N;i++){
    for(int j=0;j<SENSORS_N;j++){
      printf("%lf,",challenger.cw1[i][j] );
      fprintf(wrlog,"%lf,",challenger.cw1[i][j] );
    }
  }
  printf("][");
  fprintf(wrlog, "][");
  for(int i=0;i<OUTPUT_N;i++){
    for(int j=0;j<HIDDEN_N;j++){
      printf("%lf,",challenger.cw2[i][j] );
      fprintf(wrlog,"%lf,",challenger.cw2[i][j] );
    }
  }
  printf("]]\n");
  fprintf(wrlog, "]]\n");
}





