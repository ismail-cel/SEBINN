function P = Koch(n)
%Koch draws Koch snowflake
%n is how many iterations of the creative process we want 
% initialize P to an equilateral triangle 
radius = 0.45;

% Define angles for the vertices of the equilateral triangle
angles = linspace(0, 2*pi, 4); % Divide the circle into 3 equal angles

% Coordinates of the vertices
y_vertices = radius * cos(angles(1:3)); % x-coordinates
x_vertices = radius * sin(angles(1:3)); % y-coordinates

P = [x_vertices', y_vertices'; x_vertices(1), y_vertices(1)];
%P = [ 0 0; 
 %     1 0; 
  %    cos(-pi/3), sin(-pi/3); 
   %   0 0  ];

  %center
for iteration=1:n
   newP = zeros( size(P,1)*4+1, 2);%%3
   
   for i=1:size(P,1)-1
       newP(4*i+1,:) = P(i,:);
       newP(4*i+2,:) = (2*P(i,:) + P(i+1,:) )/3;
       link = P(i+1,:)-P(i,:);
       ang = atan2( link(2), link(1) );   
       linkLeng = sqrt( sum(link.^2) );   
       newP(4*i+3,:) = newP(4*i+2,:) + (linkLeng/3)*[ cos(ang+pi/3), sin(ang+pi/3) ];
       newP(4*i+4,:) = (P(i,:) + 2*P(i+1,:) )/3;
   end
   newP( 4*size(P,1)+1,:) = P(size(P,1),:);
   P = newP;
   P= P(any(P,2),:) ;
   %P = [0 0;P;0 0];
end
% now join up the points in P
   %clf;        % clear the figure window   
  plot( P(:,1), P(:,2) ) ; % plot P
  axis equal
   %axis equal; % make the x- and y-scale the same
   P = P(1:end-1,:);
end